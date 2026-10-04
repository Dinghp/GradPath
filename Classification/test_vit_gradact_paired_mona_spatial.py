from __future__ import annotations

import argparse
import importlib
import json
import os
import re
from typing import Dict, Iterable, Tuple

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def compute_metrics(labels, predictions):
    return {
        "Accuracy": float(accuracy_score(labels, predictions)),
        "Precision": float(
            precision_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "Recall": float(
            recall_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "F1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", ""}:
        return False

    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_target_blocks(value):
    if value is None:
        return "all"

    if isinstance(value, (list, tuple)):
        return [int(index) for index in value]

    value = str(value).strip()
    lowered = value.lower()

    if lowered == "all" or lowered.startswith("last"):
        return lowered

    if "," in value:
        return [
            int(index.strip())
            for index in value.split(",")
            if index.strip()
        ]

    return [int(value)]


def find_latest_experiment(method_dir: str) -> str:
    if not os.path.isdir(method_dir):
        raise FileNotFoundError(
            f"Experiment directory does not exist: {method_dir}"
        )

    experiment_dirs = [
        os.path.join(method_dir, name)
        for name in os.listdir(method_dir)
        if os.path.isdir(os.path.join(method_dir, name))
    ]

    if not experiment_dirs:
        raise FileNotFoundError(
            f"No experiment folders were found in: {method_dir}"
        )

    return max(experiment_dirs, key=os.path.getmtime)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]):
    cleaned_state = {}

    for key, value in state_dict.items():
        while key.startswith("module."):
            key = key[len("module."):]
        cleaned_state[key] = value

    return cleaned_state


def recover_selected_indices_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
):
    selected_indices = {}
    pattern = re.compile(
        r"^backbone\.blocks\.(\d+)\.mlp\."
        r"(?:fc1|fc2)\.selected_indices$"
    )

    for key, value in state_dict.items():
        match = pattern.match(key)
        if match is None:
            continue

        block_index = int(match.group(1))
        indices = torch.as_tensor(
            value,
            dtype=torch.long,
        ).cpu()

        if block_index in selected_indices:
            if not torch.equal(
                selected_indices[block_index],
                indices,
            ):
                raise RuntimeError(
                    "fc1 and fc2 selected indices differ in "
                    f"block {block_index}"
                )
        else:
            selected_indices[block_index] = indices

    return selected_indices


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[int, torch.Tensor],
    dict,
]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
        selected_indices = checkpoint.get(
            "selected_indices"
        )
        checkpoint_info = checkpoint
    else:
        state_dict = checkpoint
        selected_indices = None
        checkpoint_info = {}

    if not isinstance(state_dict, dict):
        raise TypeError(
            "The checkpoint does not contain a valid state_dict"
        )

    state_dict = strip_module_prefix(state_dict)

    if selected_indices is None:
        selected_indices = recover_selected_indices_from_state_dict(
            state_dict
        )
    else:
        selected_indices = {
            int(block_index): torch.as_tensor(
                indices,
                dtype=torch.long,
            ).cpu()
            for block_index, indices in selected_indices.items()
        }

    if not selected_indices:
        raise RuntimeError(
            "No selected FFN neuron indices were found in the "
            "checkpoint. The same paths must be injected before "
            "loading the state_dict."
        )

    return state_dict, selected_indices, checkpoint_info


def infer_update_selected_bias(
    state_dict: Dict[str, torch.Tensor],
    config: dict,
    checkpoint_info: dict,
) -> bool:
    # state_dict is the most reliable source because strict loading must
    # reproduce the exact module structure saved during training.
    state_has_selected_bias = any(
        key.endswith(".selected_bias")
        for key in state_dict
    )

    checkpoint_args = checkpoint_info.get("args", {})
    configured_value = config.get(
        "update_selected_bias",
        checkpoint_args.get("update_selected_bias", None),
    )

    if configured_value is not None:
        configured_bias = parse_bool(configured_value)
        if configured_bias != state_has_selected_bias:
            print(
                "Warning: update_selected_bias in config/checkpoint "
                "does not match the state_dict. The state_dict "
                f"structure will be used: {state_has_selected_bias}."
            )

    return state_has_selected_bias


def parse_spatial_kernels(value):
    if isinstance(value, (tuple, list)):
        kernels = tuple(
            int(kernel)
            for kernel in value
        )
    else:
        kernels = tuple(
            int(item.strip())
            for item in str(value).split(",")
            if item.strip()
        )

    if not kernels:
        kernels = (3, 5, 7)

    return kernels


def infer_spatial_settings(
    state_dict: Dict[str, torch.Tensor],
    config: dict,
    checkpoint_info: dict,
):
    checkpoint_args = checkpoint_info.get(
        "args",
        {},
    )

    state_has_spatial = any(
        ".spatial_conv." in key
        for key in state_dict
    )

    configured_use = config.get(
        "use_spatial",
        checkpoint_args.get(
            "use_spatial",
            None,
        ),
    )

    if configured_use is None:
        use_spatial = state_has_spatial
    else:
        use_spatial = parse_bool(
            configured_use
        )

        if use_spatial != state_has_spatial:
            print(
                "Warning: use_spatial in config/checkpoint does not "
                "match the state_dict. The state_dict structure will "
                f"be used: {state_has_spatial}."
            )
            use_spatial = state_has_spatial

    spatial_dim = int(
        config.get(
            "spatial_dim",
            checkpoint_args.get(
                "spatial_dim",
                0,
            ),
        )
    )

    spatial_kernels = parse_spatial_kernels(
        config.get(
            "spatial_kernels",
            checkpoint_args.get(
                "spatial_kernels",
                "3,5,7",
            ),
        )
    )

    spatial_gamma_init = float(
        config.get(
            "spatial_gamma_init",
            checkpoint_args.get(
                "spatial_gamma_init",
                -3.0,
            ),
        )
    )

    return (
        use_spatial,
        spatial_dim,
        spatial_kernels,
        spatial_gamma_init,
    )


def build_gradact_model(
    model_module: str,
    num_classes: int,
    target_blocks,
    selected_indices: Dict[int, torch.Tensor],
    update_selected_bias: bool,
    use_spatial: bool,
    spatial_dim: int,
    spatial_kernels,
    spatial_gamma_init: float,
):
    if model_module.endswith(".py"):
        model_module = model_module[:-3]

    module = importlib.import_module(model_module)

    if not hasattr(module, "VIT_GradActPaired"):
        raise AttributeError(
            f"{model_module} does not contain VIT_GradActPaired"
        )

    model_class = getattr(module, "VIT_GradActPaired")
    model = model_class(
        num_classes=num_classes,
        weights_path=None,
        target_blocks=target_blocks,
    )

    model.inject_from_indices(
        selected_indices=selected_indices,
        update_selected_bias=update_selected_bias,
        use_spatial=use_spatial,
        spatial_dim=spatial_dim,
        spatial_kernels=spatial_kernels,
        spatial_gamma_init=spatial_gamma_init,
    )

    return model


def resolve_dataset_root(
    command_line_root,
    config: dict,
    dataset: str,
) -> str:
    if command_line_root is not None:
        return command_line_root

    configured_root = config.get("dataset_root")
    if configured_root:
        return str(configured_root)

    return os.path.join("dataset", dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test the configurable VIT GradAct paired-path model "
            "using the best or last checkpoint"
        )
    )
    parser.add_argument(
        "--method",
        type=str,
        default="GradActPaired_MonaSpatial",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="DermaMNIST",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="run_files_GradActPaired_MonaSpatial",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help=(
            "Exact experiment folder. When omitted, the latest "
            "folder under exp_dir/method/dataset is used."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_best.pth",
    )
    parser.add_argument(
        "--model_module",
        type=str,
        default="train_vit_gradact_paired_mona_spatial",
        help=(
            "Importable Python module containing VIT_GradActPaired. "
            "Use the filename without .py."
        ),
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    if args.run_dir is not None:
        experiment_dir = args.run_dir
    else:
        method_dir = os.path.join(
            args.exp_dir,
            args.method,
            args.dataset,
        )
        experiment_dir = find_latest_experiment(method_dir)

    config_path = os.path.join(
        experiment_dir,
        "config.json",
    )
    checkpoint_path = os.path.join(
        experiment_dir,
        args.checkpoint,
    )

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Config file was not found: {config_path}"
        )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint was not found: {checkpoint_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    num_classes = int(config["num_classes"])
    target_blocks = parse_target_blocks(
        config.get("target_blocks", "all")
    )
    dataset_root = resolve_dataset_root(
        command_line_root=args.dataset_root,
        config=config,
        dataset=args.dataset,
    )
    test_dir = os.path.join(dataset_root, "test")

    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Test directory does not exist: {test_dir}"
        )

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]
    )

    test_set = datasets.ImageFolder(
        test_dir,
        transform=transform,
    )

    if len(test_set.classes) != num_classes:
        raise ValueError(
            f"The test set contains {len(test_set.classes)} classes "
            f"{test_set.classes}, but the checkpoint was trained "
            f"with num_classes={num_classes}"
        )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(
            max(args.num_workers, 0),
            os.cpu_count() or 1,
        ),
        pin_memory=torch.cuda.is_available(),
    )

    (
        state_dict,
        selected_indices,
        checkpoint_info,
    ) = load_checkpoint(checkpoint_path)

    update_selected_bias = infer_update_selected_bias(
        state_dict=state_dict,
        config=config,
        checkpoint_info=checkpoint_info,
    )

    (
        use_spatial,
        spatial_dim,
        spatial_kernels,
        spatial_gamma_init,
    ) = infer_spatial_settings(
        state_dict=state_dict,
        config=config,
        checkpoint_info=checkpoint_info,
    )

    net = build_gradact_model(
        model_module=args.model_module,
        num_classes=num_classes,
        target_blocks=target_blocks,
        selected_indices=selected_indices,
        update_selected_bias=update_selected_bias,
        use_spatial=use_spatial,
        spatial_dim=spatial_dim,
        spatial_kernels=spatial_kernels,
        spatial_gamma_init=spatial_gamma_init,
    )

    net.load_state_dict(state_dict, strict=True)
    net.sync_selected_weights()

    # Merge selected rows/columns/bias back into ordinary Linear layers.
    # This removes training-only copies while preserving the same output.
    net.merge_for_inference()
    net.to(device)
    net.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(
            test_loader,
            desc=f"Testing {args.dataset}",
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            outputs = net(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            predictions = outputs.argmax(dim=1)
            all_predictions.extend(
                predictions.cpu().tolist()
            )
            all_labels.extend(labels.tolist())

    metrics = compute_metrics(
        all_labels,
        all_predictions,
    )

    result = {
        "Method": args.method,
        "Dataset": args.dataset,
        "Experiment": os.path.basename(experiment_dir),
        "ExperimentDirectory": experiment_dir,
        "Checkpoint": args.checkpoint,
        "CheckpointEpoch": checkpoint_info.get("epoch"),
        "CheckpointValidationAccuracy": checkpoint_info.get(
            "validation_accuracy"
        ),
        "UpdateSelectedBias": update_selected_bias,
        "UseSpatial": use_spatial,
        "SpatialDim": spatial_dim,
        "SpatialKernels": list(spatial_kernels),
        "SpatialGammaInit": spatial_gamma_init,
        "Classes": test_set.classes,
        "ClassToIndex": test_set.class_to_idx,
        "TestSamples": len(test_set),
        "SelectedRankByBlock": {
            str(block_index): int(indices.numel())
            for block_index, indices in selected_indices.items()
        },
        **metrics,
    }

    checkpoint_stem = os.path.splitext(
        os.path.basename(args.checkpoint)
    )[0]
    result_path = os.path.join(
        experiment_dir,
        f"test_result_{checkpoint_stem}.json",
    )

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"\n[VIT_GradActPaired | {args.dataset}] Test Results"
    )
    print(f"Experiment: {experiment_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Update selected bias: {update_selected_bias}")
    print(
        "Spatial: use={} | dim={} | kernels={} | gamma_init={}".format(
            use_spatial,
            spatial_dim,
            spatial_kernels,
            spatial_gamma_init,
        )
    )
    print("Selected ranks:", result["SelectedRankByBlock"])

    for metric_name in (
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ):
        print(
            f"{metric_name}: {result[metric_name]:.4f}"
        )

    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
