# type: ignore

"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
kd_utilities.py

Utilities for YOLO-Pose Feature-Level Knowledge Distillation.

Contains:
    - tensor/logging helpers
    - batch device helper
    - feature hook manager
    - feature KD loss
    - KD criterion wrapper
    - custom PoseFeatureKDTrainer

    
Author: Giorgia Gobbi (GIOG)
E-mail: ggobbi@indra.es
Creation Date: 2026-05-01

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""


import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from ultralytics import YOLO
from ultralytics.models.yolo.pose.train import PoseTrainer
import sys 

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from pipeline.dvd_use_case.small_model.utilities.read_yaml import read_yaml

# ═══════════════════════════════════════════════════════════════
# Config helpers
# ═══════════════════════════════════════════════════════════════

def get_kd_config_path() -> Path:
    """
    Read KD config path from environment.

    The training script should set:

        os.environ["POSE_KD_CONFIG_PATH"] = str(CONFIG_PATH)

    before calling model.train(...).
    """
    return Path(os.environ.get("POSE_KD_CONFIG_PATH", "config_kd.yaml"))


def load_kd_training_config() -> tuple[dict, dict, Path]:
    """
    Load full YAML config and return:

        training_config, kd_config, config_path
    """
    config_path = get_kd_config_path()

    if not config_path.exists():
        raise FileNotFoundError(
            f"KD config file not found: {config_path}. "
            "Set os.environ['POSE_KD_CONFIG_PATH'] in train_kd.py."
        )

    config = read_yaml(config_path)

    training_config = config.get("training", {})
    kd_config = config.get("kd", {})

    return training_config, kd_config, config_path


# ═══════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════

def tensor_to_float(x: Any, reduce: str = "sum") -> float:
    """
    Convert tensors or numeric values to Python float for logging.

    If tensor has multiple values:
        reduce="sum"  -> sum all elements
        reduce="mean" -> average all elements
    """
    if isinstance(x, torch.Tensor):
        x = x.detach()

        if x.numel() == 1:
            return float(x.cpu())

        if reduce == "sum":
            return float(x.sum().cpu())

        if reduce == "mean":
            return float(x.mean().cpu())

        raise ValueError(f"Unknown reduce mode: {reduce}")

    return float(x)


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """
    Recursively move every tensor in a nested Ultralytics batch to device.
    Strings, paths, image filenames, shapes, etc. are unchanged.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)

    if isinstance(batch, dict):
        return {
            key: move_batch_to_device(value, device)
            for key, value in batch.items()
        }

    if isinstance(batch, list):
        return [
            move_batch_to_device(value, device)
            for value in batch
        ]

    if isinstance(batch, tuple):
        return tuple(
            move_batch_to_device(value, device)
            for value in batch
        )

    return batch


# ═══════════════════════════════════════════════════════════════
# Feature Hook Manager
# ═══════════════════════════════════════════════════════════════

class FeatureHookManager:
    """
    Registers forward hooks on selected YOLO layers and stores feature maps.

    For Ultralytics YOLO models, the actual layer list is usually:

        model.model

    Example layer indices:
        [-5, -4, -3]
        [-4, -3, -2]
        [15, 18, 21]
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: list[int],
        detach: bool = False,
        name: str = "model",
    ):
        self.model = model
        self.layer_indices = layer_indices
        self.detach = detach
        self.name = name

        self.features: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

        self._register_hooks()

    def _get_layer(self, index: int) -> nn.Module:
        if not hasattr(self.model, "model"):
            raise RuntimeError(
                f"{self.name} has no attribute .model. "
                "Cannot register YOLO layer hooks."
            )

        layers = self.model.model

        if index < 0:
            index = len(layers) + index

        if index < 0 or index >= len(layers):
            raise IndexError(
                f"Invalid layer index {index} for {self.name}. "
                f"Model has {len(layers)} layers."
            )

        return layers[index]

    @staticmethod
    def _extract_tensor(output: Any) -> torch.Tensor | None:
        """
        Extract the first usable tensor from hook output.
        """
        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, (list, tuple)):
            for item in output:
                tensor = FeatureHookManager._extract_tensor(item)
                if tensor is not None:
                    return tensor

        if isinstance(output, dict):
            for item in output.values():
                tensor = FeatureHookManager._extract_tensor(item)
                if tensor is not None:
                    return tensor

        return None

    def _make_hook(self, layer_idx: int):
        def hook(module: nn.Module, inputs: Any, output: Any):
            tensor = self._extract_tensor(output)

            if tensor is None:
                return

            # We only use convolutional feature maps.
            if tensor.ndim != 4:
                return

            if self.detach:
                tensor = tensor.detach()

            self.features[layer_idx] = tensor

        return hook

    def _register_hooks(self):
        for idx in self.layer_indices:
            layer = self._get_layer(idx)
            handle = layer.register_forward_hook(self._make_hook(idx))
            self.handles.append(handle)

    def clear(self):
        self.features.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()
        self.features.clear()


# ═══════════════════════════════════════════════════════════════
# Feature KD Loss
# ═══════════════════════════════════════════════════════════════

class FeatureKDLoss(nn.Module):
    """
    Feature-level KD using attention transfer.

    Instead of directly matching channels, which is difficult when teacher and
    student have different widths, it computes a spatial attention map:

        attention = mean(feature^2 over channels)

    This produces [B, 1, H, W] maps for both teacher and student.
    """

    def __init__(
        self,
        mode: str = "attention",
        normalize: bool = True,
        layer_weights: dict[int, float] | None = None,
        verbose: bool = True,
    ):
        super().__init__()

        self.mode = mode
        self.normalize = normalize
        self.layer_weights = layer_weights or {}
        self.verbose = verbose

        self._warned_empty = False

    def _attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert [B, C, H, W] feature map to [B, 1, H, W] attention map.
        """
        attn = x.pow(2).mean(dim=1, keepdim=True)

        if self.normalize:
            b = attn.shape[0]
            flat = attn.reshape(b, -1)
            flat = F.normalize(flat, p=2, dim=1, eps=1e-6)
            attn = flat.reshape_as(attn)

        return attn

    def _feature_mse_crop(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        """
        Direct feature MSE with channel cropping.

        Less robust than attention mode, but useful for experiments.
        """
        min_c = min(student.shape[1], teacher.shape[1])

        student = student[:, :min_c]
        teacher = teacher[:, :min_c]

        if self.normalize:
            student = F.normalize(student, p=2, dim=1, eps=1e-6)
            teacher = F.normalize(teacher, p=2, dim=1, eps=1e-6)

        return F.mse_loss(student, teacher)

    def forward(
        self,
        student_features: dict[int, torch.Tensor],
        teacher_features: dict[int, torch.Tensor],
    ) -> torch.Tensor:

        common_layers = sorted(
            set(student_features.keys()).intersection(set(teacher_features.keys()))
        )

        if not common_layers:
            if self.verbose and not self._warned_empty:
                print(
                    "[FeatureKD WARNING] No common hooked features found. "
                    "Feature KD loss will be zero."
                )
                self._warned_empty = True

            if student_features:
                first = next(iter(student_features.values()))
                return first.sum() * 0.0

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return torch.zeros((), device=device)

        total_loss = None
        total_weight = 0.0

        for layer_idx in common_layers:
            student = student_features[layer_idx]
            teacher = teacher_features[layer_idx].detach()

            if student.ndim != 4 or teacher.ndim != 4:
                continue

            # Match spatial resolution.
            if student.shape[-2:] != teacher.shape[-2:]:
                teacher = F.interpolate(
                    teacher,
                    size=student.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if self.mode == "attention":
                student_kd = self._attention_map(student)
                teacher_kd = self._attention_map(teacher)
                layer_loss = F.mse_loss(student_kd, teacher_kd)

            elif self.mode == "mse_crop":
                layer_loss = self._feature_mse_crop(student, teacher)

            else:
                raise ValueError(f"Unknown FeatureKDLoss mode: {self.mode}")

            weight = float(self.layer_weights.get(layer_idx, 1.0))

            if total_loss is None:
                total_loss = weight * layer_loss
            else:
                total_loss = total_loss + weight * layer_loss

            total_weight += weight

        if total_loss is None:
            if student_features:
                first = next(iter(student_features.values()))
                return first.sum() * 0.0

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return torch.zeros((), device=device)

        return total_loss / max(total_weight, 1e-6)


# ═══════════════════════════════════════════════════════════════
# Feature KD Criterion Wrapper
# ═══════════════════════════════════════════════════════════════

class FeatureKDPoseCriterion:
    """
    Wraps the original Ultralytics pose criterion and adds feature-level KD.

    This object is attached to:

        self.model.criterion

    Therefore it must be safe for EMA deepcopy.

    Important:
        - Do not store trainer_ref here.
        - Do not store dataloaders here.
        - __getstate__ removes teacher/hooks from EMA deepcopy.
    """

    def __init__(
        self,
        base_criterion: Any,
        teacher_model: nn.Module,
        feature_kd_loss_fn: FeatureKDLoss,
        kd_weight: float,
        feature_layers: list[int],
        warmup_epochs: int = 0,
        decay: bool = False,
        total_epochs: int = 1,
    ):
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model

        self.feature_kd_loss_fn = feature_kd_loss_fn

        self.base_kd_weight = float(kd_weight)
        self.kd_weight = float(kd_weight)

        self.feature_layers = feature_layers

        self.warmup_epochs = int(warmup_epochs)
        self.decay = bool(decay)
        self.total_epochs = max(int(total_epochs), 1)
        self.current_epoch = 0

        # Hooks are attached lazily after EMA is created.
        self.student_hooks: FeatureHookManager | None = None
        self.teacher_hooks: FeatureHookManager | None = None
        self._hooks_attached = False

    def __getstate__(self):
        """
        Prevent EMA deepcopy from copying teacher model and hooks.
        """
        state = self.__dict__.copy()

        state["teacher_model"] = None
        state["student_hooks"] = None
        state["teacher_hooks"] = None
        state["_hooks_attached"] = False

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def attach_hooks(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
    ):
        """
        Attach hooks after EMA has been created.

        This avoids deepcopy problems and avoids EMA copying hook closures.
        """
        if self._hooks_attached:
            return

        self.teacher_model = teacher_model

        self.student_hooks = FeatureHookManager(
            model=student_model,
            layer_indices=self.feature_layers,
            detach=False,
            name="student",
        )

        self.teacher_hooks = FeatureHookManager(
            model=teacher_model,
            layer_indices=self.feature_layers,
            detach=True,
            name="teacher",
        )

        self._hooks_attached = True

        print("=" * 60)
        print("Feature KD hooks attached")
        print(f"Feature layers: {self.feature_layers}")
        print("=" * 60)

    def clear_features(self):
        if self.student_hooks is not None:
            self.student_hooks.clear()

        if self.teacher_hooks is not None:
            self.teacher_hooks.clear()

    def set_epoch(self, epoch: int):
        """
        Update KD weight with optional warmup/decay.
        """
        self.current_epoch = int(epoch)

        weight = self.base_kd_weight

        if self.warmup_epochs > 0 and self.current_epoch < self.warmup_epochs:
            weight *= float(self.current_epoch + 1) / float(self.warmup_epochs)

        if self.decay:
            start = max(self.warmup_epochs, 0)

            if self.current_epoch > start:
                denom = max(self.total_epochs - start, 1)
                progress = min(
                    max((self.current_epoch - start) / denom, 0.0),
                    1.0,
                )
                weight *= 1.0 - progress

        self.kd_weight = float(weight)

    def __call__(
        self,
        preds: Any,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if isinstance(batch, dict) and isinstance(batch.get("img"), torch.Tensor):
            device = batch["img"].device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch = move_batch_to_device(batch, device)

        # Original Ultralytics supervised YOLO-Pose loss.
        supervised_loss, loss_items = self.base_criterion(preds, batch)

        if isinstance(supervised_loss, torch.Tensor) and supervised_loss.numel() > 1:
            supervised_loss = supervised_loss.sum()

        # If this is EMA copy or hooks are unavailable, return supervised only.
        if (
            self.teacher_model is None
            or self.student_hooks is None
            or self.teacher_hooks is None
        ):
            return supervised_loss, loss_items

        imgs = batch["img"]

        # Student features were captured during the student forward pass.
        student_features = dict(self.student_hooks.features)

        # Teacher forward to capture teacher features.
        self.teacher_hooks.clear()
        self.teacher_model.to(device)
        self.teacher_model.eval()

        with torch.no_grad():
            _ = self.teacher_model(imgs)

        teacher_features = dict(self.teacher_hooks.features)

        feature_kd_loss = self.feature_kd_loss_fn(
            student_features=student_features,
            teacher_features=teacher_features,
        )

        if isinstance(feature_kd_loss, torch.Tensor) and feature_kd_loss.numel() > 1:
            feature_kd_loss = feature_kd_loss.sum()

        total_loss = supervised_loss + self.kd_weight * feature_kd_loss

        if wandb.run is not None:
            log_dict = {
                "train/supervised_loss": tensor_to_float(supervised_loss, reduce="sum"),
                "train/feature_kd_loss": tensor_to_float(feature_kd_loss, reduce="sum"),
                "train/kd_weight": float(self.kd_weight),
                "train/total_loss": tensor_to_float(total_loss, reduce="sum"),
            }

            if isinstance(loss_items, torch.Tensor):
                detached_items = loss_items.detach().cpu().flatten()

                for i, value in enumerate(detached_items):
                    log_dict[f"train/loss_item_{i}"] = float(value)

            wandb.log(log_dict)

        return total_loss, loss_items


# ═══════════════════════════════════════════════════════════════
# Custom Feature KD Trainer
# ═══════════════════════════════════════════════════════════════

class PoseFeatureKDTrainer(PoseTrainer):
    """
    YOLO-Pose trainer with feature-level teacher-student KD.
    """

    def setup_model(self):
        """
        Only prepare KD config here.

        Do not call self.model.init_criterion() here because model/loss device
        can be wrong at this stage.
        """
        ckpt = super().setup_model()

        (
            self._training_config,
            self._kd_config,
            self._config_path,
        ) = load_kd_training_config()

        self._kd_enabled = bool(self._kd_config.get("enabled", True))
        self._kd_wrapped_model_criterion = False
        self._feature_hooks_attached = False

        if not self._kd_enabled:
            print("=" * 60)
            print("Feature Knowledge Distillation disabled from config.")
            print(f"Config path: {self._config_path}")
            print("=" * 60)
            return ckpt

        self.kd_weight = float(self._kd_config.get("weight", 0.05))
        self.kd_warmup_epochs = int(self._kd_config.get("warmup_epochs", 20))
        self.kd_decay = bool(self._kd_config.get("decay", False))

        print("=" * 60)
        print("Feature Knowledge Distillation config loaded")
        print(f"Config path       : {self._config_path}")
        print(f"KD enabled        : {self._kd_enabled}")
        print(f"KD weight         : {self.kd_weight}")
        print(f"KD warmup epochs  : {self.kd_warmup_epochs}")
        print(f"KD decay          : {self.kd_decay}")
        print("=" * 60)

        return ckpt

    def set_model_attributes(self):
        """
        Attach feature-level KD criterion after Ultralytics sets model attrs.

        Hooks are not attached here. They are attached lazily in preprocess_batch,
        after EMA has already been created.
        """
        super().set_model_attributes()

        if not getattr(self, "_kd_enabled", False):
            return

        if not hasattr(self.model, "args"):
            self.model.args = self.args

        self.model = self.model.to(self.device)

        teacher_weights = self._kd_config.get("teacher_weights")
        teacher_weights = str(teacher_weights)

        if not Path(teacher_weights).exists():
            raise FileNotFoundError(
                f"Teacher weights not found: {teacher_weights}"
            )

        self.teacher_yolo = YOLO(teacher_weights)
        self.teacher = self.teacher_yolo.model.to(self.device)
        self.teacher.eval()

        for p in self.teacher.parameters():
            p.requires_grad = False

        feature_layers = self._kd_config.get("feature_layers", [-5, -4, -3])

        if isinstance(feature_layers, tuple):
            feature_layers = list(feature_layers)

        feature_layers = [int(i) for i in feature_layers]

        raw_layer_weights = self._kd_config.get("feature_layer_weights", None)
        layer_weights = None

        if isinstance(raw_layer_weights, dict):
            layer_weights = {
                int(k): float(v)
                for k, v in raw_layer_weights.items()
            }

        feature_kd_loss_fn = FeatureKDLoss(
            mode=str(self._kd_config.get("feature_mode", "attention")),
            normalize=bool(self._kd_config.get("feature_normalize", True)),
            layer_weights=layer_weights,
            verbose=bool(self._kd_config.get("verbose", True)),
        )

        # Build base Ultralytics pose loss after model is on correct device.
        base_criterion = self.model.init_criterion()
        self._force_loss_device(base_criterion, self.device)

        self.model.criterion = FeatureKDPoseCriterion(
            base_criterion=base_criterion,
            teacher_model=self.teacher,
            feature_kd_loss_fn=feature_kd_loss_fn,
            kd_weight=self.kd_weight,
            feature_layers=feature_layers,
            warmup_epochs=self.kd_warmup_epochs,
            decay=self.kd_decay,
            total_epochs=int(
                self._training_config.get("epochs", getattr(self, "epochs", 1))
            ),
        )

        self._kd_wrapped_model_criterion = True

        student_params = sum(p.numel() for p in self.model.parameters())
        teacher_params = sum(p.numel() for p in self.teacher.parameters())

        print("=" * 60)
        print("Feature Knowledge Distillation enabled")
        print(f"Teacher weights      : {teacher_weights}")
        print(f"Feature layers       : {feature_layers}")
        print(f"Feature KD mode      : {feature_kd_loss_fn.mode}")
        print(f"Feature normalize    : {feature_kd_loss_fn.normalize}")
        print(f"Student params       : {student_params:,}")
        print(f"Teacher params       : {teacher_params:,}")
        print(f"KD weight            : {self.kd_weight}")
        print(f"KD warmup epochs     : {self.kd_warmup_epochs}")
        print(f"KD decay             : {self.kd_decay}")
        print(f"Wrapped criterion    : {self._kd_wrapped_model_criterion}")
        print("=" * 60)

    def _force_loss_device(self, loss_obj: Any, device: torch.device):
        """
        Recursively patch common Ultralytics loss objects to use the correct device.
        """
        visited = set()

        def patch(obj: Any):
            obj_id = id(obj)

            if obj_id in visited:
                return

            visited.add(obj_id)

            if hasattr(obj, "device"):
                try:
                    obj.device = device
                except Exception:
                    pass

            for attr in [
                "one2many",
                "one2one",
                "assigner",
                "assigner2",
                "bce",
                "hyp",
            ]:
                if hasattr(obj, attr):
                    try:
                        patch(getattr(obj, attr))
                    except Exception:
                        pass

        patch(loss_obj)

    def preprocess_batch(self, batch: dict) -> dict:
        """
        Ultralytics calls this before self.model(batch).

        We use it to:
            - move all batch tensors to device
            - update KD epoch schedule
            - lazily attach hooks after EMA creation
            - clear feature buffers before forward
        """
        batch = super().preprocess_batch(batch)
        batch = move_batch_to_device(batch, self.device)

        criterion = getattr(self.model, "criterion", None)

        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(int(getattr(self, "epoch", 0)))

        # Attach hooks lazily here, not during set_model_attributes.
        # This avoids EMA deepcopy issues.
        if (
            hasattr(criterion, "attach_hooks")
            and not getattr(criterion, "_hooks_attached", False)
        ):
            criterion.attach_hooks(
                student_model=self.model,
                teacher_model=self.teacher,
            )

        if hasattr(criterion, "clear_features"):
            criterion.clear_features()

        return batch