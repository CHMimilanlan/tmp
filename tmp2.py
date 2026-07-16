import contextlib
import math
import os
import re
import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch
import wandb
import yaml
from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs, gather_object, set_seed
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    StepLR,
)
from torch.utils.data import DataLoader

from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.config_display import print_config
from ltx_trainer.datasets import PrecomputedDataset
from ltx_trainer.gpu_utils import free_gpu_memory, get_gpu_memory_gb
from ltx_trainer.hf_hub_utils import push_to_hub
from ltx_trainer.model_loader import load_embeddings_processor, load_transformer
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.quantization import quantize_model
from ltx_trainer.sigma_tracker import SigmaBucketTracker
from ltx_trainer.timestep_samplers import SAMPLERS
from ltx_trainer.training_state import ConfigFingerprint, RngStates, TrainingState
from ltx_trainer.training_strategies import get_training_strategy
from ltx_trainer.validation_runner import ValidationRunner
from ltx_core.model.transformer.spatial_track_encoder import *

# Disable irrelevant warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Silence bitsandbytes warnings about casting
warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

# Disable progress bars if not main process
IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"
if not IS_MAIN_PROCESS:
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()

StepCallback = Callable[[int, int, list[Path]], None]  # (step, total, list[sampled_video_path]) -> None

MEMORY_CHECK_INTERVAL = 200

import re
from pathlib import Path

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper
from safetensors.torch import load_file
from torch import Tensor

MOTION_TRACK_ADAPTER = "motion_track"
AV2AV_ADAPTER = "av2av"

class TrainingStats(BaseModel):
    """Statistics collected during training"""

    total_time_seconds: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float
    global_batch_size: int
    num_processes: int


@dataclass(frozen=True)
class TrainingStepOutput:
    """Output from a single training step."""

    loss: Tensor  # [B,] per-element loss (unreduced)
    sigma: Tensor  # [B,] sampled sigma, detached from computational graph


class LtxvTrainer:
    def __init__(self, trainer_config: LtxTrainerConfig) -> None:
        self._config = trainer_config
        if IS_MAIN_PROCESS:
            print_config(trainer_config)
        self._training_strategy = get_training_strategy(self._config.training_strategy)
        # ValidationRunner loads its own models (text encoder, VAE encoder/decoder, etc.),
        # caches prompt embeddings and conditioning media, then unloads encoders.
        self._validation_runner = ValidationRunner(
            config=self._config.validation,
            model_path=self._config.model.model_path,
            text_encoder_path=self._config.model.text_encoder_path,
            load_text_encoder_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
        )

        self._load_models()
        self._setup_accelerator()
        self._collect_trainable_params()
        self._loaded_checkpoint_path: Path | None = None
        self._load_checkpoint()
        self._prepare_models_for_training()
        self._dataset = None
        self._global_step = -1
        self._checkpoint_paths: list[Path] = []
        self._training_state_paths: list[Path] = []
        self._training_state_size_warned = False
        self._sigma_tracker = SigmaBucketTracker()
        self._wandb_run = None
        
    def train(  # noqa: PLR0912, PLR0915
        self,
        disable_progress_bars: bool = False,
        step_callback: StepCallback | None = None,
    ) -> tuple[Path, TrainingStats]:
        """
        Start the training process.
        Args:
            disable_progress_bars: Disable Rich progress bars (useful for multi-process runs).
            step_callback: Optional callback invoked after each optimization step.
        Returns:
            Tuple of (saved_model_path, training_stats)
        """
        device = self._accelerator.device
        cfg = self._config
        start_mem = get_gpu_memory_gb(device)

        train_start_time = time.time()

        initial_step, training_state = self._resume_state
        resuming = training_state is not None

        set_seed(cfg.seed)
        logger.debug(f"Process {self._accelerator.process_index} using seed: {cfg.seed}")

        self._init_optimizer()

        if training_state is not None and not self._restore_training_state(training_state):
            initial_step = 0
            resuming = False

        # Initialize W&B after restore so we only resume the run when state restore succeeds.
        resume_run_id = training_state.wandb_run_id if resuming and training_state is not None else None
        self._init_wandb(resume_run_id=resume_run_id)

        self._init_dataloader()
        data_iter = iter(self._dataloader)
        self._init_timestep_sampler()

        # Synchronize all processes after initialization
        self._accelerator.wait_for_everyone()

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # Save the training configuration as YAML
        self._save_config()

        remaining_steps = cfg.optimization.steps - initial_step
        if remaining_steps <= 0:
            raise ValueError(
                f"No remaining training steps: initial_step={initial_step} >= "
                f"target_steps={cfg.optimization.steps}. Nothing to train."
            )

        if resuming:
            logger.info(f"🚀 Resuming training from step {initial_step} → {cfg.optimization.steps}")
        else:
            logger.info("🚀 Starting training...")

        # Create progress tracking (disabled for non-main processes or when explicitly disabled)
        progress_enabled = IS_MAIN_PROCESS and not disable_progress_bars
        # progress_enabled = False
        progress = TrainingProgress(
            enabled=progress_enabled,
            total_steps=remaining_steps,
        )

        if IS_MAIN_PROCESS and disable_progress_bars:
            logger.warning("Progress bars disabled. Intermediate status messages will be logged instead.")

        self._transformer.train()
        self._global_step = initial_step

        peak_mem_during_training = start_mem

        sampled_videos_paths = None
        with progress:
            if cfg.validation.interval and not cfg.validation.skip_initial_validation:
                with self._offloaded_optimizer_state():
                    sampled_videos_paths = self._run_validation(progress)

            self._accelerator.wait_for_everyone()

            for step in range(remaining_steps * cfg.optimization.gradient_accumulation_steps):
                # Get next batch, reset the dataloader if needed
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self._dataloader)
                    batch = next(data_iter)

                step_start_time = time.time()
                with self._accelerator.accumulate(self._transformer):
                    is_optimization_step = (step + 1) % cfg.optimization.gradient_accumulation_steps == 0
                    if is_optimization_step:
                        self._global_step += 1
                        
                    output = self._training_step(batch)
                    self._accelerator.backward(output.loss.mean())

                    # =========================================================
                    if (
                        self._global_step == 1
                        and self._accelerator.sync_gradients
                        and self._accelerator.is_main_process
                    ):
                        av2av_with_grad = []
                        spatial_with_grad = []
                        unexpected_with_grad = []

                        for name, param in (
                            self._transformer.named_parameters()
                        ):
                            if param.grad is None:
                                continue

                            grad_norm = (
                                param.grad
                                .detach()
                                .float()
                                .norm()
                                .item()
                            )

                            if (
                                "lora_A.default" in name
                                or "lora_B.default" in name
                            ):
                                av2av_with_grad.append(
                                    (name, grad_norm)
                                )

                            elif (
                                "spatial_track_encoder."
                                "modules_to_save.default" in name
                            ):
                                spatial_with_grad.append(
                                    (name, grad_norm)
                                )

                            else:
                                unexpected_with_grad.append(
                                    (name, grad_norm)
                                )

                        logger.info(
                            "\nFirst-step gradient audit:\n"
                            f"  AV2AV LoRA tensors with grad: "
                            f"{len(av2av_with_grad)}\n"
                            f"  SpatialTrack tensors with grad: "
                            f"{len(spatial_with_grad)}\n"
                            f"  Unexpected tensors with grad: "
                            f"{len(unexpected_with_grad)}"
                        )

                        if not av2av_with_grad:
                            raise RuntimeError(
                                "AV2AV LoRA did not receive gradients."
                            )

                        if not spatial_with_grad:
                            raise RuntimeError(
                                "SpatialTrackEncoder did not receive gradients. "
                                "Check no_grad(), detach(), and whether its "
                                "output contributes to the final loss."
                            )

                        if unexpected_with_grad:
                            raise RuntimeError(
                                "Unexpected frozen parameters received gradients:\n"
                                + "\n".join(
                                    name
                                    for name, _ in unexpected_with_grad[:100]
                                )
                            )
                    # =========================================================


                    if self._accelerator.sync_gradients and cfg.optimization.max_grad_norm > 0:
                        self._accelerator.clip_grad_norm_(
                            self._trainable_params,
                            cfg.optimization.max_grad_norm,
                        )

                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    # Run validation if needed (handles DDP/FSDP work distribution internally)
                    if (
                        cfg.validation.interval
                        and self._global_step > 0
                        and self._global_step % cfg.validation.interval == 0
                        and is_optimization_step
                    ):
                        with self._offloaded_optimizer_state():
                            sampled_videos_paths = self._run_validation(progress)

                    # Save checkpoint if needed
                    if (
                        cfg.checkpoints.interval
                        and self._global_step > 0
                        and self._global_step % cfg.checkpoints.interval == 0
                        and is_optimization_step
                    ):
                        self._save_checkpoint()

                    self._accelerator.wait_for_everyone()

                    # Call step callback if provided
                    if step_callback and is_optimization_step:
                        step_callback(self._global_step, cfg.optimization.steps, sampled_videos_paths)

                    self._accelerator.wait_for_everyone()

                    # Update progress and log metrics
                    current_lr = self._optimizer.param_groups[0]["lr"]
                    step_time = (time.time() - step_start_time) * cfg.optimization.gradient_accumulation_steps
                    step_loss = output.loss.detach().mean().item()

                    progress.update_training(
                        loss=step_loss,
                        lr=current_lr,
                        step_time=step_time,
                        advance=is_optimization_step,
                    )

                    # Log metrics to W&B (only on main process and optimization steps)
                    if IS_MAIN_PROCESS and is_optimization_step:
                        # Track per-element loss by sigma bucket
                        self._sigma_tracker.update(output.sigma.cpu().tolist(), output.loss.detach().cpu().tolist())
                        metrics = {
                            "train/loss": step_loss,
                            "train/learning_rate": current_lr,
                            "train/step_time": step_time,
                            "train/global_step": self._global_step,
                        }
                        metrics.update(self._sigma_tracker.get_metrics())
                        self._log_metrics(metrics)

                    # Fallback logging when progress bars are disabled
                    if disable_progress_bars and IS_MAIN_PROCESS and self._global_step % 20 == 0:
                        elapsed = time.time() - train_start_time
                        steps_done = self._global_step - initial_step
                        if steps_done > 0:
                            total_estimated = elapsed / steps_done * remaining_steps
                            total_time = f"{total_estimated // 3600:.0f}h {(total_estimated % 3600) // 60:.0f}m"
                        else:
                            total_time = "calculating..."
                        logger.info(
                            f"Step {self._global_step}/{cfg.optimization.steps} - "
                            f"Loss: {step_loss:.4f}, LR: {current_lr:.2e}, "
                            f"Time/Step: {step_time:.2f}s, Total Time: {total_time}",
                        )

                    # Sample GPU memory periodically
                    if step % MEMORY_CHECK_INTERVAL == 0:
                        current_mem = get_gpu_memory_gb(device)
                        peak_mem_during_training = max(peak_mem_during_training, current_mem)

        # Collect final stats
        train_end_time = time.time()
        end_mem = get_gpu_memory_gb(device)
        peak_mem = max(start_mem, end_mem, peak_mem_during_training)

        # Calculate steps/second over entire training
        total_time_seconds = train_end_time - train_start_time
        steps_per_second = remaining_steps / total_time_seconds

        samples_per_second = steps_per_second * self._accelerator.num_processes * cfg.optimization.batch_size

        stats = TrainingStats(
            total_time_seconds=total_time_seconds,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            peak_gpu_memory_gb=peak_mem,
            num_processes=self._accelerator.num_processes,
            global_batch_size=cfg.optimization.batch_size * self._accelerator.num_processes,
        )

        saved_path = self._save_checkpoint()

        if IS_MAIN_PROCESS:
            # Log the training statistics
            self._log_training_stats(stats)

            # Upload artifacts to hub if enabled
            if cfg.hub.push_to_hub:
                push_to_hub(saved_path, sampled_videos_paths, self._config)

            # Log final stats to W&B
            if self._wandb_run is not None:
                self._log_metrics(
                    {
                        "stats/total_time_minutes": stats.total_time_seconds / 60,
                        "stats/steps_per_second": stats.steps_per_second,
                        "stats/samples_per_second": stats.samples_per_second,
                        "stats/peak_gpu_memory_gb": stats.peak_gpu_memory_gb,
                    }
                )
                self._wandb_run.finish()

        self._accelerator.wait_for_everyone()
        self._accelerator.end_training()

        return saved_path, stats

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> TrainingStepOutput:
        """Perform a single training step using the configured strategy."""
        # Apply embedding connectors to transform pre-computed text embeddings
        conditions = batch["conditions"]
        if "video_prompt_embeds" in conditions:
            # New format: separate video/audio features from precompute()
            video_features = conditions["video_prompt_embeds"]
            audio_features = conditions.get("audio_prompt_embeds")
        else:
            # Legacy format: single prompt_embeds tensor — duplicate for both modalities
            video_features = conditions["prompt_embeds"]
            audio_features = conditions["prompt_embeds"]

        mask = conditions["prompt_attention_mask"]
        additive_mask = convert_to_additive_mask(mask, video_features.dtype)
        video_embeds, audio_embeds, attention_mask = self._embeddings_processor.create_embeddings(
            video_features, audio_features, additive_mask
        )

        conditions["video_prompt_embeds"] = video_embeds
        conditions["audio_prompt_embeds"] = audio_embeds
        conditions["prompt_attention_mask"] = attention_mask

        # Use strategy to prepare training inputs (returns ModelInputs with Modality objects)
        # model_inputs = self._training_strategy.prepare_training_inputs(batch, self._timestep_sampler, self._transformer)
        model_inputs = self._training_strategy.prepare_customize_training_inputs(batch, self._timestep_sampler, self._transformer)

        # Run transformer forward pass with Modality-based interface
        video_pred, audio_pred = self._transformer(
            video=model_inputs.video,
            audio=model_inputs.audio,
            perturbations=None,
        )

        # Use strategy to compute loss (returns per-element [B,] for sigma-bucket tracking)
        loss = self._training_strategy.compute_loss(video_pred, audio_pred, model_inputs)

        # Sigma comes from whichever modality is generated (video preferred, else audio).
        if model_inputs.video is not None and model_inputs.video.enabled:
            sigma = model_inputs.video.sigma.detach()
        else:
            sigma = model_inputs.audio.sigma.detach()

        return TrainingStepOutput(loss=loss, sigma=sigma)

    def _load_models(self) -> None:
        """Load the transformer and embeddings processor for training."""
        logger.debug("Loading transformer...")
        self._transformer = load_transformer(
            checkpoint_path=self._config.model.model_path,
            device="cpu",
            dtype=torch.bfloat16,
        )

        # DDP-safe: LOCAL_RANK is set by accelerate before trainer init. Loading on bare
        # "cuda" would resolve to cuda:0 on every rank and crash with a device mismatch.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        init_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        logger.debug("Loading embeddings processor...")
        self._embeddings_processor = load_embeddings_processor(
            checkpoint_path=self._config.model.model_path,
            device=init_device,
            dtype=torch.bfloat16,
        )
        self._embeddings_processor.feature_extractor = None

        transformer_dtype = torch.bfloat16 if self._config.model.training_mode == "lora" else torch.float32
        self._transformer = self._transformer.to(dtype=transformer_dtype)

        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")

            logger.info(f'Quantizing model with "{self._config.acceleration.quantization}". This may take a while...')
            self._transformer = quantize_model(
                self._transformer,
                precision=self._config.acceleration.quantization,
            )

        self._transformer.requires_grad_(False)

        # ---------------------------------------------------------
        # Initialize custom modules after loading/freezing LTX model
        # ---------------------------------------------------------
        spatial_cfg = SpatialTrackEncoderConfig(
            dim=128,
            video_t=16,
            video_h=8,
            video_w=12,
            audio_t=122,
            num_heads=8,
            dropout=0.0,
            encoder_depth=4,
            decoder_depth=2,
        )

        self._transformer.initialize_spatial_track_modules(
            encoder_type="audio_query",
            cfg=spatial_cfg,
            device=torch.device("cpu"),
            dtype=transformer_dtype,
        )

    # def _collect_trainable_params(self) -> None:
    #     """Collect trainable parameters based on training mode."""
    #     if self._config.model.training_mode == "lora":
    #         # For LoRA training, first set up LoRA layers
    #         self._setup_lora()
    #     elif self._config.model.training_mode == "full":
    #         # For full training, unfreeze all transformer parameters
    #         self._transformer.requires_grad_(True)
    #     else:
    #         raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")

    #     self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]
    #     logger.debug(f"Trainable params count: {sum(p.numel() for p in self._trainable_params):,}")

    def _collect_trainable_params(self) -> None:
        """Collect trainable AV2AV and SpatialTrack parameters."""
        if self._config.model.training_mode == "lora":
            self._setup_lora()

        elif self._config.model.training_mode == "full":
            self._transformer.requires_grad_(True)

        else:
            raise ValueError(
                f"Unknown training mode: "
                f"{self._config.model.training_mode}"
            )

        self._trainable_params = [
            param
            for param in self._transformer.parameters()
            if param.requires_grad
        ]

        motion_total_numel = 0
        motion_trainable_numel = 0
        av2av_trainable_numel = 0
        spatial_trainable_numel = 0
        unexpected_trainable_numel = 0

        for name, param in (
            self._transformer.named_parameters()
        ):
            is_motion = (
                f".lora_A.{MOTION_TRACK_ADAPTER}." in name
                or
                f".lora_B.{MOTION_TRACK_ADAPTER}." in name
            )

            is_av2av = (
                f".lora_A.{AV2AV_ADAPTER}." in name
                or
                f".lora_B.{AV2AV_ADAPTER}." in name
            )

            is_spatial = (
                "spatial_track_encoder."
                f"modules_to_save.{AV2AV_ADAPTER}."
                in name
            )

            if is_motion:
                motion_total_numel += param.numel()

                if param.requires_grad:
                    motion_trainable_numel += (
                        param.numel()
                    )

            elif is_av2av and param.requires_grad:
                av2av_trainable_numel += param.numel()

            elif is_spatial and param.requires_grad:
                spatial_trainable_numel += param.numel()

            elif param.requires_grad:
                unexpected_trainable_numel += (
                    param.numel()
                )
                logger.warning(
                    f"Unexpected trainable parameter: {name}"
                )

        logger.info(
            "\nOptimizer parameter summary:\n"
            f"  Motion-track total:     "
            f"{motion_total_numel:,}\n"
            f"  Motion-track trainable: "
            f"{motion_trainable_numel:,}\n"
            f"  AV2AV LoRA:             "
            f"{av2av_trainable_numel:,}\n"
            f"  SpatialTrackEncoder:    "
            f"{spatial_trainable_numel:,}\n"
            f"  Unexpected:             "
            f"{unexpected_trainable_numel:,}\n"
            f"  Optimizer total:        "
            f"{sum(p.numel() for p in self._trainable_params):,}"
        )

        if motion_total_numel == 0:
            raise RuntimeError(
                "Motion-track LoRA parameters are missing."
            )

        if motion_trainable_numel != 0:
            raise RuntimeError(
                "Motion-track LoRA was included in optimizer."
            )

        if av2av_trainable_numel == 0:
            raise RuntimeError(
                "AV2AV LoRA has no trainable parameters."
            )

        if spatial_trainable_numel == 0:
            raise RuntimeError(
                "SpatialTrackEncoder has no trainable "
                "parameters."
            )

        if unexpected_trainable_numel != 0:
            raise RuntimeError(
                "Unexpected parameters will enter optimizer."
            )

    def _init_timestep_sampler(self) -> None:
        """Initialize the timestep sampler based on the config."""
        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

    # def _setup_lora(self) -> None:
    #     """Configure LoRA adapters for the transformer. Only called in LoRA training mode."""
    #     logger.debug(f"Adding LoRA adapter with rank {self._config.lora.rank}")
    #     # lora_config = LoraConfig(
    #     #     r=self._config.lora.rank,
    #     #     lora_alpha=self._config.lora.alpha,
    #     #     target_modules=self._config.lora.target_modules,
    #     #     lora_dropout=self._config.lora.dropout,
    #     #     init_lora_weights=True,
    #     # )
        
    #     lora_config = LoraConfig(
    #         r=self._config.lora.rank,
    #         lora_alpha=self._config.lora.alpha,
    #         target_modules=self._config.lora.target_modules,
    #         lora_dropout=self._config.lora.dropout,
    #         init_lora_weights=True,

    #         # Train and save these complete newly-added modules.
    #         modules_to_save=[
    #             "spatial_track_encoder",
    #         ],
    #     )
        
    #     # Wrap the transformer with PEFT to add LoRA layers
    #     # noinspection PyTypeChecker
    #     self._transformer = get_peft_model(self._transformer, lora_config)

    #     audit_lora_model(self._transformer)
    #     # ---------------------------------------------------------
    #     # 增加一次检查
    #     # ---------------------------------------------------------
    #     base_model = self._transformer.get_base_model()

    #     assert base_model.spatial_track_encoder is not None
    #     trainable_custom_params = sum(
    #         p.numel()
    #         for p in base_model.spatial_track_encoder.parameters()
    #         if p.requires_grad
    #     )

    #     logger.info(
    #         f"Trainable SpatialTrackEncoder params: "
    #         f"{trainable_custom_params:,}"
    #     )

    #     if trainable_custom_params == 0:
    #         raise RuntimeError(
    #             "SpatialTrackEncoder is frozen after PEFT wrapping."
    #         )

    def _setup_lora(self) -> None:
        """
        Configure two simultaneously active LoRA adapters:

        1. motion_track:
        - loaded from pretrained Motion-track IC-LoRA
        - active in forward
        - frozen

        2. av2av:
        - newly initialized or resumed from AV2AV checkpoint
        - active in forward
        - trainable

        SpatialTrackEncoder:
        - belongs to AV2AV adapter through modules_to_save
        - trainable
        """
        logger.info(
            "Setting up frozen Motion-track LoRA + "
            "trainable AV2AV LoRA."
        )

        cfg = self._config.lora

        if cfg.motion_track_checkpoint is None:
            raise ValueError(
                "lora.motion_track_checkpoint must be set."
            )

        # =========================================================
        # 1. 读取并定位 Motion-track LoRA
        # =========================================================
        (
            motion_state_dict,
            motion_rank,
            motion_target_modules,
        ) = _inspect_motion_track_checkpoint(
            cfg.motion_track_checkpoint
        )

        motion_alpha = (
            cfg.motion_track_alpha
            if cfg.motion_track_alpha is not None
            else motion_rank
        )

        logger.info(
            "Creating Motion-track adapter:\n"
            f"  name: {MOTION_TRACK_ADAPTER}\n"
            f"  checkpoint: {cfg.motion_track_checkpoint}\n"
            f"  rank: {motion_rank}\n"
            f"  alpha: {motion_alpha}\n"
            f"  target module count: "
            f"{len(motion_target_modules)}"
        )

        # =========================================================
        # 2. 创建 Motion-track adapter
        # =========================================================
        motion_lora_config = LoraConfig(
            r=motion_rank,
            lora_alpha=motion_alpha,

            # 使用 checkpoint 中解析出来的精确层名
            target_modules=motion_target_modules,

            lora_dropout=0.0,
            init_lora_weights=True,
            inference_mode=True,
            bias="none",

            # 非常重要：
            # Motion-track adapter 不管理 SpatialTrackEncoder
            modules_to_save=None,
        )

        self._transformer = get_peft_model(
            self._transformer,
            motion_lora_config,
            adapter_name=MOTION_TRACK_ADAPTER,
        )

        # =========================================================
        # 3. 将预训练权重加载到 motion_track adapter
        # =========================================================
        load_result = set_peft_model_state_dict(
            self._transformer,
            motion_state_dict,
            adapter_name=MOTION_TRACK_ADAPTER,
        )

        unexpected_keys = list(
            getattr(load_result, "unexpected_keys", [])
        )

        missing_motion_keys = [
            key
            for key in getattr(
                load_result,
                "missing_keys",
                [],
            )
            if (
                "lora_" in key
                and MOTION_TRACK_ADAPTER in key
            )
        ]

        if unexpected_keys:
            raise RuntimeError(
                "Unexpected Motion-track checkpoint keys:\n"
                + "\n".join(unexpected_keys[:100])
            )

        if missing_motion_keys:
            raise RuntimeError(
                "Some Motion-track LoRA parameters were not "
                "loaded:\n"
                + "\n".join(missing_motion_keys[:100])
            )

        logger.info(
            "Motion-track LoRA checkpoint loaded."
        )

        # =========================================================
        # 4. 创建新的 AV2AV adapter
        # =========================================================
        av2av_lora_config = LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            target_modules=cfg.target_modules,
            lora_dropout=cfg.dropout,
            init_lora_weights=True,
            inference_mode=False,
            bias="none",

            # 只有 AV2AV adapter 管理并训练 SpatialTrackEncoder
            modules_to_save=[
                "spatial_track_encoder",
            ],
        )

        self._transformer.add_adapter(
            adapter_name=AV2AV_ADAPTER,
            peft_config=av2av_lora_config,
        )

        # =========================================================
        # 5. 先冻结所有参数
        # =========================================================
        self._transformer.requires_grad_(False)

        # =========================================================
        # 6. 同时激活 Motion-track 和 AV2AV adapter
        # =========================================================
        #
        # set_adapter 会把两个 LoRA 都设置为 active，
        # 使两者同时参与 forward。
        #
        # inference_mode=False 会暂时将两个 adapter 都设为 trainable，
        # 所以下面必须重新单独冻结 Motion-track。
        self._transformer.set_adapter(
            [
                MOTION_TRACK_ADAPTER,
                AV2AV_ADAPTER,
            ],
            inference_mode=False,
        )

        # =========================================================
        # 7. 冻结 Motion-track，开启 AV2AV 梯度
        # =========================================================
        self._transformer.set_requires_grad(
            MOTION_TRACK_ADAPTER,
            requires_grad=False,
        )

        self._transformer.set_requires_grad(
            AV2AV_ADAPTER,
            requires_grad=True,
        )

        # =========================================================
        # 8. 严格检查最终训练状态
        # =========================================================
        self._audit_frozen_motion_trainable_av2av()

        audit_lora_model(
            self._transformer
        )
    

    def _load_checkpoint(self) -> None:
        """Load checkpoint if specified in config, then resolve resume state."""
        if not self._config.model.load_checkpoint:
            self._resume_state: tuple[int, TrainingState | None] = (0, None)
            return

        checkpoint_path = self._find_checkpoint(self._config.model.load_checkpoint)
        if not checkpoint_path:
            logger.warning(f"⚠️ Could not find checkpoint at {self._config.model.load_checkpoint}")
            self._resume_state = (0, None)
            return

        self._loaded_checkpoint_path = checkpoint_path
        logger.info(f"📥 Loading checkpoint from {checkpoint_path}")

        if self._config.model.training_mode == "full":
            self._load_full_checkpoint(checkpoint_path)
        else:  # LoRA mode
            self._load_lora_checkpoint(checkpoint_path)

        self._resume_state = self._resolve_resume_state()

    def _load_full_checkpoint(self, checkpoint_path: Path) -> None:
        """Load full model checkpoint."""
        state_dict = load_file(checkpoint_path)
        self._transformer.load_state_dict(state_dict, strict=True)

        logger.info("✅ Full model checkpoint loaded successfully")

    def _load_lora_checkpoint(self, checkpoint_path: Path) -> None:
        """Load LoRA checkpoint with DDP/FSDP compatibility."""
        state_dict = load_file(checkpoint_path)

        # Adjust layer names to match internal format.
        # (Weights are saved in ComfyUI-compatible format, with "diffusion_model." prefix)
        state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}

        # Load LoRA weights and verify all weights were loaded
        base_model = self._transformer.get_base_model()
        set_peft_model_state_dict(base_model, state_dict)

        logger.info("✅ LoRA checkpoint loaded successfully")

    def _resolve_resume_state(self) -> tuple[int, TrainingState | None]:
        """Determine resume state by looking for a training state file next to the loaded checkpoint.
        Returns (initial_step, TrainingState or None).
        If no_resume config is set, no checkpoint loaded, or no state file found: returns (0, None).
        """
        if self._config.checkpoints.no_resume or self._loaded_checkpoint_path is None:
            return 0, None

        state = self._load_training_state(self._loaded_checkpoint_path)
        if state is None:
            return 0, None

        fp = state.config_fingerprint
        cfg = self._config
        mismatches: list[str] = []
        if fp.optimizer_type != cfg.optimization.optimizer_type:
            mismatches.append(f"optimizer_type: {fp.optimizer_type} → {cfg.optimization.optimizer_type}")
        if fp.scheduler_type != cfg.optimization.scheduler_type:
            mismatches.append(f"scheduler_type: {fp.scheduler_type} → {cfg.optimization.scheduler_type}")
        if fp.training_mode != cfg.model.training_mode:
            mismatches.append(f"training_mode: {fp.training_mode} → {cfg.model.training_mode}")
        if (
            cfg.model.training_mode == "lora"
            and cfg.lora is not None
            and fp.lora_rank is not None
            and fp.lora_rank != cfg.lora.rank
        ):
            mismatches.append(f"lora_rank: {fp.lora_rank} → {cfg.lora.rank}")
        if mismatches:
            logger.warning(
                f"⚠️ Training state config mismatch ({', '.join(mismatches)}). "
                "Starting from step 0. Set checkpoints.no_resume=true to silence this warning."
            )
            return 0, None

        if state.global_step < 0:
            logger.warning(f"⚠️ Training state has invalid global_step={state.global_step!r}. Starting from step 0.")
            return 0, None
        logger.info(f"📌 Resuming from step {state.global_step}")
        return state.global_step, state

    @staticmethod
    def _load_training_state(checkpoint_path: Path) -> TrainingState | None:
        """Load training state file that corresponds to a checkpoint weights file."""
        match = re.search(r"step_(\d+)", checkpoint_path.name)
        if not match:
            return None

        step_str = match.group(1)
        state_path = checkpoint_path.parent / f"training_state_step_{step_str}.pt"

        if not state_path.exists():
            return None

        try:
            raw: dict = torch.load(state_path, map_location="cpu", weights_only=False)
            state = TrainingState.from_save_dict(raw)
            logger.info(f"📥 Loaded training state from {state_path}")
            return state
        except Exception as e:
            logger.warning(f"⚠️ Failed to load training state from {state_path}: {e}. Starting from step 0.")
            return None

    def _restore_training_state(self, training_state: TrainingState) -> bool:
        """Restore optimizer, scheduler, and RNG states from a loaded TrainingState.
        Must be called after _init_optimizer() (which calls accelerator.prepare).
        Returns True if restore succeeded, False if it failed (caller should fall back to step 0).
        """
        try:
            if training_state.optimizer_state_dict is not None:
                self._optimizer.load_state_dict(training_state.optimizer_state_dict)
                logger.debug("Restored optimizer state (full mode)")

            if training_state.lr_scheduler_state_dict is not None and self._lr_scheduler is not None:
                self._lr_scheduler.load_state_dict(training_state.lr_scheduler_state_dict)
                logger.debug("Restored LR scheduler state")
        except Exception as e:
            logger.warning(f"⚠️ Failed to restore training state: {e}. Starting from step 0.")
            return False

        rng = training_state.rng_states
        if self._accelerator.num_processes > 1:
            logger.debug("Skipping RNG restore in multi-process mode (only main process state was saved)")
        else:
            if rng.torch_state is not None:
                torch.random.set_rng_state(rng.torch_state)
            if rng.cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(rng.cuda_state)
            logger.debug("Restored RNG states")

        return True

    def _prepare_models_for_training(self) -> None:
        """Prepare models for training with Accelerate."""

        # For FSDP + LoRA: Cast entire model to FP32.
        # FSDP requires uniform dtype across all parameters in wrapped modules.
        # In LoRA mode, PEFT creates LoRA params in FP32 while base model is BF16.
        # We cast the base model to FP32 to match the LoRA params.
        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.model.training_mode == "lora":
            logger.debug("FSDP: casting transformer to FP32 for uniform dtype")
            self._transformer = self._transformer.to(dtype=torch.float32)

        # Enable gradient checkpointing if requested
        # For PeftModel, we need to access the underlying base model
        transformer = (
            self._transformer.get_base_model() if hasattr(self._transformer, "get_base_model") else self._transformer
        )

        transformer.set_gradient_checkpointing(self._config.optimization.enable_gradient_checkpointing)

        # noinspection PyTypeChecker
        self._transformer = self._accelerator.prepare(self._transformer)
        
        initialize_fsdp_hierarchy(self._transformer)
        
        # Log GPU memory usage after model preparation
        vram_usage_gb = torch.cuda.memory_allocated() / 1024**3
        logger.debug(f"GPU memory usage after models preparation: {vram_usage_gb:.2f} GB")

    @staticmethod
    def _find_checkpoint(checkpoint_path: str | Path) -> Path | None:
        """Find the checkpoint file to load, handling both file and directory paths."""
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file():
            if not checkpoint_path.suffix == ".safetensors":
                raise ValueError(f"Checkpoint file must have a .safetensors extension: {checkpoint_path}")
            return checkpoint_path

        if checkpoint_path.is_dir():
            # Look for checkpoint files in the directory
            checkpoints = list(checkpoint_path.rglob("*step_*.safetensors"))

            if not checkpoints:
                return None

            # Sort by step number and return the latest
            def _get_step_num(p: Path) -> int:
                try:
                    return int(p.stem.split("step_")[1])
                except (IndexError, ValueError):
                    return -1

            latest = max(checkpoints, key=_get_step_num)
            return latest

        else:
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}. Must be a file or directory.")

    def _init_dataloader(self) -> None:
        """Initialize the training data loader using the strategy's data sources."""
        if self._dataset is None:
            # Get data sources from the training strategy
            data_sources = self._config.training_strategy.get_data_sources()

            self._dataset = PrecomputedDataset(self._config.data.preprocessed_data_root, data_sources=data_sources)
            logger.debug(f"Loaded dataset with {len(self._dataset):,} samples from sources: {list(data_sources)}")

        num_workers = self._config.data.num_dataloader_workers
        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
        )

        self._dataloader = self._accelerator.prepare(dataloader)

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights for the transformer."""
        logger.debug("Initializing LoRA weights...")
        for _, module in self._transformer.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.reset_lora_parameters(adapter_name="default", init_lora_weights=True)

    def _init_optimizer(self) -> None:
        """Initialize the optimizer and learning rate scheduler."""
        opt_cfg = self._config.optimization

        lr = opt_cfg.learning_rate
        if opt_cfg.optimizer_type == "adamw":
            optimizer = AdamW(self._trainable_params, lr=lr)
        elif opt_cfg.optimizer_type == "adamw8bit":
            # noinspection PyUnresolvedReferences
            from bitsandbytes.optim import AdamW8bit  # noqa: PLC0415

            optimizer = AdamW8bit(self._trainable_params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.optimizer_type}")

        lr_scheduler = self._create_scheduler(optimizer)

        # noinspection PyTypeChecker
        self._optimizer, self._lr_scheduler = self._accelerator.prepare(optimizer, lr_scheduler)

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        """Create learning rate scheduler based on config."""
        scheduler_type = self._config.optimization.scheduler_type
        steps = self._config.optimization.steps
        params = self._config.optimization.scheduler_params or {}

        if scheduler_type is None:
            return None

        if scheduler_type == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=params.pop("start_factor", 1.0),
                end_factor=params.pop("end_factor", 0.1),
                total_iters=steps,
                **params,
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=steps,
                eta_min=params.pop("eta_min", 0),
                **params,
            )
        elif scheduler_type == "cosine_with_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.pop("T_0", steps // 4),
                T_mult=params.pop("T_mult", 1),
                eta_min=params.pop("eta_min", 5e-5),
                **params,
            )
        elif scheduler_type == "polynomial":
            scheduler = PolynomialLR(
                optimizer,
                total_iters=steps,
                power=params.pop("power", 1.0),
                **params,
            )
        elif scheduler_type == "step":
            scheduler = StepLR(
                optimizer,
                step_size=params.pop("step_size", steps // 2),
                gamma=params.pop("gamma", 0.1),
                **params,
            )
        elif scheduler_type == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return scheduler

    def _setup_accelerator(self) -> None:
        """Initialize the Accelerator with the appropriate settings."""

        # find_unused_parameters=True keeps DDP happy when LoRA targets a branch the forward
        # pass skips (e.g. audio LoRA with `with_audio: false`, or short module patterns like
        # "to_k" that match the audio branch unintentionally). It's a no-op for FSDP and
        # single-GPU runs. The probing cost is paid only on the first step.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        # All distributed setup (DDP/FSDP, number of processes, etc.) is controlled by
        # the user's Accelerate configuration (accelerate config / accelerate launch).
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs],
        )

        if self._accelerator.num_processes > 1:
            logger.info(
                f"{self._accelerator.distributed_type.value} distributed training enabled "
                f"with {self._accelerator.num_processes} processes"
            )

            local_batch = self._config.optimization.batch_size
            global_batch = self._config.optimization.batch_size * self._accelerator.num_processes
            logger.info(f"Local batch size: {local_batch}, global batch size: {global_batch}")

        # Log torch.compile status from Accelerate's dynamo plugin
        is_compile_enabled = (
            hasattr(self._accelerator.state, "dynamo_plugin") and self._accelerator.state.dynamo_plugin.backend != "NO"
        )
        if is_compile_enabled:
            plugin = self._accelerator.state.dynamo_plugin
            logger.info(f"🔥 torch.compile enabled via Accelerate: backend={plugin.backend}, mode={plugin.mode}")

            if self._accelerator.distributed_type == DistributedType.FSDP:
                logger.warning(
                    "⚠️ FSDP + torch.compile is experimental and may hang on the first training iteration. "
                    "If this occurs, disable torch.compile by removing dynamo_config from your Accelerate config."
                )

        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.acceleration.quantization:
            logger.warning(
                f"FSDP with quantization ({self._config.acceleration.quantization}) may have compatibility issues."
                "Monitor training stability and consider disabling quantization if issues arise."
            )

    @contextlib.contextmanager
    def _offloaded_optimizer_state(self) -> Iterator[None]:
        """Context manager that offloads optimizer state to CPU during validation.
        Opt-in via `acceleration.offload_optimizer_during_validation`. Frees VRAM for
        validation video generation when optimizer state is large (e.g. full fine-tune
        AdamW, high-rank LoRA). No-op for FSDP (sharded state -- manual `.cpu()` breaks
        metadata).
        """
        enabled = (
            self._config.acceleration.offload_optimizer_during_validation
            and self._accelerator.distributed_type != DistributedType.FSDP
        )

        # Track exactly which tensors we move so we don't promote ones that were
        # intentionally on CPU (e.g. AdamW's `step` scalar on recent PyTorch).
        offloaded: list[tuple[dict, str]] = []
        if enabled:
            offloaded_bytes = 0
            for state in self._optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        offloaded.append((state, k))
                        offloaded_bytes += v.nbytes
            if offloaded:
                logger.info(f"Offloading optimizer state to CPU ({offloaded_bytes / 1e9:.1f} GB)")
                for state, k in offloaded:
                    state[k] = state[k].cpu()

        try:
            yield
        finally:
            device = self._accelerator.device
            for state, k in offloaded:
                state[k] = state[k].to(device)

    def _run_validation(self, progress: TrainingProgress) -> list[Path]:
        """Run distributed validation by delegating to the ValidationRunner.
        Each rank generates its assigned subset of validation samples (round-robin by
        `process_index`/`num_processes`), so all GPUs stay busy and no rank idles long
        enough to trigger NCCL timeouts. Paths are gathered across ranks so rank 0 has
        the full list for W&B logging.
        Under FSDP with multiple processes, ranks pad with extra generate passes
        (same sample, no disk write) so every rank runs the same number of forwards --
        avoids collective mismatch.
        Note: Multi-node training requires a shared filesystem so rank 0 can read
        videos written by other ranks.
        """
        self._optimizer.zero_grad(set_to_none=True)
        free_gpu_memory()

        num_samples = len(self._config.validation.samples)
        if num_samples == 0:
            return []

        rank = self._accelerator.process_index
        world_size = self._accelerator.num_processes

        rank_indices = list(range(rank, num_samples, world_size))
        work_items: list[tuple[int, bool]] = [(i, True) for i in rank_indices]
        if self._accelerator.distributed_type == DistributedType.FSDP and world_size > 1:
            # FSDP forwards run collective ops; pad short ranks with no-save duplicates so
            # every rank executes the same number of forwards. A rank with empty
            # rank_indices (world_size > num_samples) still pads with sample 0 to stay in
            # sync with the others.
            max_per_rank = math.ceil(num_samples / world_size)
            pad_seed = rank_indices[-1] if rank_indices else 0
            work_items += [(pad_seed, False)] * (max_per_rank - len(work_items))

        # W&B logging is handled by the trainer (after gathering across ranks),
        # so we always pass wandb_run=None to the runner.
        sampled = self._validation_runner.run(
            transformer=self._transformer,
            step=self._global_step,
            output_dir=Path(self._config.output_dir),
            device=self._accelerator.device,
            progress=progress,
            wandb_run=None,
            work_items=work_items,
        )

        if world_size > 1:
            sampled = sorted(gather_object(sampled), key=lambda x: x[0])

        paths = [p for _, p in sampled]

        if (
            self._accelerator.is_main_process
            and paths
            and self._config.wandb.log_validation_videos
            and self._wandb_run is not None
        ):
            self._validation_runner.log_to_wandb(self._wandb_run, paths, self._global_step)

        # Non-main ranks must not reach checkpoint collectives while main is still logging to W&B.
        self._accelerator.wait_for_everyone()

        return paths

    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        """Log training statistics."""
        stats_str = (
            "📊 Training Statistics:\n"
            f" - Total time: {stats.total_time_seconds / 60:.1f} minutes\n"
            f" - Training speed: {stats.steps_per_second:.2f} steps/second\n"
            f" - Samples/second: {stats.samples_per_second:.2f}\n"
            f" - Peak GPU memory: {stats.peak_gpu_memory_gb:.2f} GB"
        )
        if stats.num_processes > 1:
            stats_str += f"\n - Number of processes: {stats.num_processes}\n"
            stats_str += f" - Global batch size: {stats.global_batch_size}"
        logger.info(stats_str)

    def _save_checkpoint(self) -> Path | None:
        """Save the model weights."""
        is_lora = self._config.model.training_mode == "lora"
        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP

        # Prepare paths
        save_dir = Path(self._config.output_dir) / "checkpoints"
        prefix = "lora" if is_lora else "model"
        filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename

        # Get state dict (collective operation - all processes must participate)
        self._accelerator.wait_for_everyone()
        full_state_dict = self._accelerator.get_state_dict(self._transformer)

        if not IS_MAIN_PROCESS:
            return None

        save_dir.mkdir(exist_ok=True, parents=True)

        # Determine save precision
        save_dtype = torch.bfloat16 if self._config.checkpoints.precision == "bfloat16" else torch.float32

        # For LoRA: extract only adapter weights; for full: use as-is
        if is_lora:
            unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
            # For FSDP, pass full_state_dict since model params aren't directly accessible
            state_dict = get_peft_model_state_dict(
                unwrapped,
                state_dict=(
                    full_state_dict
                    if is_fsdp
                    else None
                ),
                adapter_name=AV2AV_ADAPTER,
            )

            # Remove "base_model.model." prefix added by PEFT
            state_dict = {k.replace("base_model.model.", "", 1): v for k, v in state_dict.items()}

            # Convert to ComfyUI-compatible format (add "diffusion_model." prefix)
            state_dict = {f"diffusion_model.{k}": v for k, v in state_dict.items()}

            # Cast to configured precision
            state_dict = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in state_dict.items()}

            # Build metadata for safetensors file
            metadata = self._build_checkpoint_metadata()

            # Save to disk with metadata
            save_file(state_dict, saved_weights_path, metadata=metadata)
        else:
            # Cast to configured precision
            full_state_dict = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in full_state_dict.items()}

            # Save to disk
            self._accelerator.save(full_state_dict, saved_weights_path)

        rel_path = saved_weights_path.relative_to(self._config.output_dir)
        logger.info(f"💾 {prefix.capitalize()} weights for step {self._global_step} saved in {rel_path}")

        self._checkpoint_paths.append(saved_weights_path)
        self._cleanup_checkpoints()

        self._save_training_state(save_dir)

        return saved_weights_path

    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints."""
        if 0 < self._config.checkpoints.keep_last_n < len(self._checkpoint_paths):
            checkpoints_to_remove = self._checkpoint_paths[: -self._config.checkpoints.keep_last_n]
            for old_checkpoint in checkpoints_to_remove:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")
            self._checkpoint_paths = self._checkpoint_paths[-self._config.checkpoints.keep_last_n :]

    def _save_training_state(self, save_dir: Path) -> None:
        """Save training state alongside checkpoint for resume.
        Respects checkpoints.save_training_state config:
        - "full": optimizer + scheduler + RNG + step
        - "minimal": scheduler + RNG + step only
        - "off": skip entirely
        """
        if not IS_MAIN_PROCESS:
            return

        mode = self._config.checkpoints.save_training_state
        if mode == "off":
            return

        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP

        optimizer_state = None
        if mode == "full":
            if is_fsdp:
                logger.warning(
                    "⚠️ save_training_state='full' is not supported with FSDP. "
                    "Saving 'minimal' state (scheduler + RNG only)."
                )
            else:
                optimizer_state = self._optimizer.state_dict()

        state = TrainingState(
            global_step=self._global_step,
            config_fingerprint=ConfigFingerprint(
                optimizer_type=self._config.optimization.optimizer_type,
                scheduler_type=self._config.optimization.scheduler_type,
                training_mode=self._config.model.training_mode,
                lora_rank=self._config.lora.rank if self._config.lora is not None else None,
            ),
            rng_states=RngStates(
                torch_state=torch.random.get_rng_state(),
                cuda_state=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            ),
            lr_scheduler_state_dict=self._lr_scheduler.state_dict() if self._lr_scheduler is not None else None,
            optimizer_state_dict=optimizer_state,
            wandb_run_id=self._wandb_run.id if self._wandb_run is not None else None,
        )

        state_path = save_dir / f"training_state_step_{self._global_step:05d}.pt"
        tmp_path = state_path.with_suffix(".pt.tmp")
        try:
            torch.save(state.to_save_dict(), tmp_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        tmp_path.rename(state_path)

        file_size_gb = state_path.stat().st_size / (1024**3)
        if file_size_gb > 1.0 and not self._training_state_size_warned:
            self._training_state_size_warned = True
            logger.warning(
                f"⚠️ Training state file is {file_size_gb:.1f} GB (full mode includes optimizer state). "
                f'Set checkpoints.save_training_state="minimal" to save only scheduler/RNG/step (~few KB), '
                f'or "off" to disable entirely.'
            )

        if not self._training_state_paths or self._training_state_paths[-1] != state_path:
            self._training_state_paths.append(state_path)
        self._cleanup_training_states()

        rel_path = state_path.relative_to(self._config.output_dir)
        logger.debug(f"Training state saved to {rel_path}")

    def _cleanup_training_states(self) -> None:
        """Clean up old training state files, using the same keep_last_n as checkpoints."""
        keep_n = self._config.checkpoints.keep_last_n
        if 0 < keep_n < len(self._training_state_paths):
            to_remove = self._training_state_paths[:-keep_n]
            for old_state in to_remove:
                if old_state.exists():
                    old_state.unlink()
                    logger.debug(f"Removed old training state: {old_state}")
            self._training_state_paths = self._training_state_paths[-keep_n:]

    def _build_checkpoint_metadata(self) -> dict[str, str]:
        """Build metadata dictionary for safetensors checkpoint.
        Delegates to the training strategy to get strategy-specific metadata
        that downstream inference pipelines may need.
        Returns:
            Dictionary of string key-value pairs for safetensors metadata.
            Values are converted to strings for safetensors compatibility.
        """
        raw_metadata = self._training_strategy.get_checkpoint_metadata()
        # Convert all values to strings for safetensors compatibility
        metadata = {k: str(v) for k, v in raw_metadata.items()}
        if metadata:
            logger.info(f"Saving checkpoint metadata: {metadata}")
        return metadata

    def _save_config(self) -> None:
        """Save the training configuration as a YAML file in the output directory."""
        if not IS_MAIN_PROCESS:
            return

        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)

        logger.info(f"💾 Training configuration saved to: {config_path.relative_to(self._config.output_dir)}")

    def _init_wandb(self, resume_run_id: str | None = None) -> None:
        """Initialize Weights & Biases run, resuming an existing run if its id is provided."""
        if not self._config.wandb.enabled or not IS_MAIN_PROCESS:
            self._wandb_run = None
            return

        wandb_config = self._config.wandb
        init_kwargs: dict[str, Any] = {
            "project": wandb_config.project,
            "entity": wandb_config.entity,
            "name": Path(self._config.output_dir).name,
            "tags": wandb_config.tags,
            "config": self._config.model_dump(),
        }
        if resume_run_id is not None:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "must"
        run = wandb.init(**init_kwargs)
        self._wandb_run = run

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to Weights & Biases."""
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)


    def _audit_frozen_motion_trainable_av2av(
        self,
    ) -> None:
        """
        Verify final state:

        - Motion-track LoRA exists
        - Motion-track LoRA is active and frozen
        - AV2AV LoRA exists and is trainable
        - SpatialTrackEncoder AV2AV copy is trainable
        - Base LTX parameters remain frozen
        """
        motion_params = []
        av2av_params = []
        spatial_trainable_params = []
        spatial_original_params = []
        unexpected_trainable = []

        for name, param in (
            self._transformer.named_parameters()
        ):
            is_motion_lora = (
                f".lora_A.{MOTION_TRACK_ADAPTER}." in name
                or
                f".lora_B.{MOTION_TRACK_ADAPTER}." in name
            )

            is_av2av_lora = (
                f".lora_A.{AV2AV_ADAPTER}." in name
                or
                f".lora_B.{AV2AV_ADAPTER}." in name
            )

            is_spatial_trainable_copy = (
                "spatial_track_encoder."
                f"modules_to_save.{AV2AV_ADAPTER}."
                in name
            )

            is_spatial_original = (
                "spatial_track_encoder.original_module."
                in name
            )

            if is_motion_lora:
                motion_params.append((name, param))
                continue

            if is_av2av_lora:
                av2av_params.append((name, param))
                continue

            if is_spatial_trainable_copy:
                spatial_trainable_params.append(
                    (name, param)
                )
                continue

            if is_spatial_original:
                spatial_original_params.append(
                    (name, param)
                )
                continue

            if param.requires_grad:
                unexpected_trainable.append(
                    (name, param)
                )

        # ---------------------------------------------------------
        # Motion-track LoRA 必须存在
        # ---------------------------------------------------------
        if not motion_params:
            raise RuntimeError(
                "No Motion-track LoRA parameters were found. "
                "The Motion-track adapter may not have been "
                "created or loaded correctly."
            )

        motion_trainable = [
            name
            for name, param in motion_params
            if param.requires_grad
        ]

        if motion_trainable:
            raise RuntimeError(
                "Motion-track LoRA should be frozen, but these "
                "parameters remain trainable:\n"
                + "\n".join(motion_trainable[:100])
            )

        # ---------------------------------------------------------
        # AV2AV LoRA 必须存在且全部可训练
        # ---------------------------------------------------------
        if not av2av_params:
            raise RuntimeError(
                "No AV2AV LoRA parameters were found."
            )

        av2av_frozen = [
            name
            for name, param in av2av_params
            if not param.requires_grad
        ]

        if av2av_frozen:
            raise RuntimeError(
                "Some AV2AV LoRA parameters are frozen:\n"
                + "\n".join(av2av_frozen[:100])
            )

        # ---------------------------------------------------------
        # SpatialTrackEncoder 副本必须可训练
        # ---------------------------------------------------------
        if not spatial_trainable_params:
            raise RuntimeError(
                "No trainable AV2AV SpatialTrackEncoder copy "
                "was found."
            )

        spatial_frozen = [
            name
            for name, param
            in spatial_trainable_params
            if not param.requires_grad
        ]

        if spatial_frozen:
            raise RuntimeError(
                "Some SpatialTrackEncoder AV2AV parameters "
                "are frozen:\n"
                + "\n".join(spatial_frozen[:100])
            )

        # ---------------------------------------------------------
        # original_module 应该冻结
        # ---------------------------------------------------------
        spatial_original_trainable = [
            name
            for name, param
            in spatial_original_params
            if param.requires_grad
        ]

        if spatial_original_trainable:
            raise RuntimeError(
                "SpatialTrackEncoder original_module should "
                "be frozen:\n"
                + "\n".join(
                    spatial_original_trainable[:100]
                )
            )

        # ---------------------------------------------------------
        # 不应有其他可训练基础参数
        # ---------------------------------------------------------
        if unexpected_trainable:
            raise RuntimeError(
                "Unexpected base-model parameters are "
                "trainable:\n"
                + "\n".join(
                    name
                    for name, _ in unexpected_trainable[:100]
                )
            )

        motion_total_numel = sum(
            param.numel()
            for _, param in motion_params
        )

        motion_trainable_numel = sum(
            param.numel()
            for _, param in motion_params
            if param.requires_grad
        )

        av2av_trainable_numel = sum(
            param.numel()
            for _, param in av2av_params
            if param.requires_grad
        )

        spatial_trainable_numel = sum(
            param.numel()
            for _, param
            in spatial_trainable_params
            if param.requires_grad
        )

        active_adapters = list(
            self._transformer.active_adapters
        )

        logger.info(
            "\nFinal multi-adapter trainability audit:\n"
            f"  Active adapters: "
            f"{active_adapters}\n"
            f"  Motion-track total params: "
            f"{motion_total_numel:,}\n"
            f"  Motion-track trainable params: "
            f"{motion_trainable_numel:,}\n"
            f"  AV2AV LoRA trainable params: "
            f"{av2av_trainable_numel:,}\n"
            f"  SpatialTrackEncoder trainable params: "
            f"{spatial_trainable_numel:,}\n"
            f"  Unexpected trainable params: "
            f"{sum(p.numel() for _, p in unexpected_trainable):,}"
        )

        expected_active = {
            MOTION_TRACK_ADAPTER,
            AV2AV_ADAPTER,
        }

        if set(active_adapters) != expected_active:
            raise RuntimeError(
                "Active adapters are incorrect:\n"
                f"  expected={sorted(expected_active)}\n"
                f"  actual={sorted(active_adapters)}"
            )

        if motion_trainable_numel != 0:
            raise RuntimeError(
                "Motion-track LoRA is not fully frozen."
            )

        logger.info(
            "Multi-adapter audit passed: Motion-track is "
            "active and frozen; AV2AV and SpatialTrackEncoder "
            "are trainable."
        )


    def _audit_av2av_and_spatial_track_encoder(
        self,
    ) -> None:
        """
        Verify:
        - motion-track adapter no longer exists as LoRA tensors
        - AV2AV/default LoRA is trainable
        - SpatialTrackEncoder trainable copy is trainable
        - SpatialTrackEncoder original copy is frozen
        - no unexpected base-model parameters are trainable
        """
        base_model = self._transformer.get_base_model()

        if not hasattr(
            base_model,
            "spatial_track_encoder",
        ):
            raise RuntimeError(
                "SpatialTrackEncoder is not registered on "
                "the underlying transformer."
            )

        spatial_wrapper = (
            base_model.spatial_track_encoder
        )

        if not isinstance(
            spatial_wrapper,
            ModulesToSaveWrapper,
        ):
            raise RuntimeError(
                "SpatialTrackEncoder was not wrapped by "
                "ModulesToSaveWrapper. Check that "
                'modules_to_save=["spatial_track_encoder"] '
                "matches the actual attribute name."
            )

        if AV2AV_ADAPTER not in (
            spatial_wrapper.modules_to_save
        ):
            raise RuntimeError(
                "No default modules_to_save copy exists for "
                "SpatialTrackEncoder."
            )

        original_encoder = (
            spatial_wrapper.original_module
        )

        trainable_encoder = (
            spatial_wrapper.modules_to_save[
                AV2AV_ADAPTER
            ]
        )

        original_trainable = [
            name
            for name, param
            in original_encoder.named_parameters()
            if param.requires_grad
        ]

        if original_trainable:
            raise RuntimeError(
                "SpatialTrackEncoder original_module should be "
                "frozen, but these parameters are trainable:\n"
                + "\n".join(original_trainable[:100])
            )

        frozen_trainable_copy = [
            name
            for name, param
            in trainable_encoder.named_parameters()
            if not param.requires_grad
        ]

        if frozen_trainable_copy:
            raise RuntimeError(
                "Some parameters in the trainable "
                "SpatialTrackEncoder copy are frozen:\n"
                + "\n".join(frozen_trainable_copy[:100])
            )

        spatial_trainable_numel = sum(
            param.numel()
            for param in trainable_encoder.parameters()
            if param.requires_grad
        )

        av2av_lora_params = []
        motion_lora_params = []
        unexpected_trainable = []

        for name, param in (
            self._transformer.named_parameters()
        ):
            if MOTION_TRACK_ADAPTER in name:
                motion_lora_params.append(
                    (name, param)
                )
                continue

            if (
                "lora_A.default" in name
                or "lora_B.default" in name
            ):
                av2av_lora_params.append(
                    (name, param)
                )
                continue

            if (
                "spatial_track_encoder."
                "modules_to_save.default" in name
            ):
                continue

            if param.requires_grad:
                unexpected_trainable.append(name)

        if motion_lora_params:
            raise RuntimeError(
                "Motion-track LoRA tensors should no longer "
                "exist after merging:\n"
                + "\n".join(
                    name
                    for name, _ in motion_lora_params[:100]
                )
            )

        frozen_av2av_lora = [
            name
            for name, param in av2av_lora_params
            if not param.requires_grad
        ]

        if not av2av_lora_params:
            raise RuntimeError(
                "No trainable AV2AV/default LoRA parameters "
                "were found."
            )

        if frozen_av2av_lora:
            raise RuntimeError(
                "Some AV2AV LoRA parameters are frozen:\n"
                + "\n".join(frozen_av2av_lora[:100])
            )

        if spatial_trainable_numel == 0:
            raise RuntimeError(
                "SpatialTrackEncoder has zero trainable "
                "parameters."
            )

        if unexpected_trainable:
            raise RuntimeError(
                "Unexpected base-model parameters are trainable:\n"
                + "\n".join(unexpected_trainable[:100])
            )

        av2av_numel = sum(
            param.numel()
            for _, param in av2av_lora_params
            if param.requires_grad
        )

        logger.info(
            "\nFinal trainability audit:\n"
            f"  AV2AV LoRA trainable params: "
            f"{av2av_numel:,}\n"
            f"  SpatialTrackEncoder trainable params: "
            f"{spatial_trainable_numel:,}\n"
            f"  Motion-track LoRA params remaining: "
            f"{len(motion_lora_params)}\n"
            f"  Unexpected trainable params: "
            f"{len(unexpected_trainable)}"
        )




from collections import Counter

import torch
from peft.tuners.tuners_utils import BaseTunerLayer


def audit_lora_model(
    model: torch.nn.Module,
    *,
    max_print: int = 300,
) -> None:
    """
    Inspect:
    1. Which modules were replaced by PEFT LoRA layers.
    2. Which LoRA parameters are trainable/frozen.
    3. Whether non-LoRA parameters are unexpectedly trainable.
    """
    lora_module_names: list[str] = []

    for module_name, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            lora_module_names.append(module_name)

    trainable_params: list[tuple[str, torch.nn.Parameter]] = []
    frozen_params: list[tuple[str, torch.nn.Parameter]] = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param))
        else:
            frozen_params.append((name, param))

    trainable_lora = [
        (name, param)
        for name, param in trainable_params
        if "lora_" in name
    ]

    frozen_lora = [
        (name, param)
        for name, param in frozen_params
        if "lora_" in name
    ]

    trainable_non_lora = [
        (name, param)
        for name, param in trainable_params
        if "lora_" not in name
    ]

    total_numel = sum(
        param.numel()
        for param in model.parameters()
    )

    trainable_numel = sum(
        param.numel()
        for _, param in trainable_params
    )

    trainable_lora_numel = sum(
        param.numel()
        for _, param in trainable_lora
    )

    print("\n" + "=" * 100)
    print("LoRA AUDIT")
    print("=" * 100)

    print(f"LoRA wrapped module count: {len(lora_module_names)}")
    print(f"Total parameter count:      {total_numel:,}")
    print(f"Trainable parameter count:  {trainable_numel:,}")
    print(f"Trainable LoRA parameters:  {trainable_lora_numel:,}")

    if total_numel > 0:
        ratio = 100.0 * trainable_numel / total_numel
        print(f"Trainable ratio:            {ratio:.6f}%")

    print("\n[LoRA wrapped modules]")

    for name in lora_module_names[:max_print]:
        print(f"  {name}")

    if len(lora_module_names) > max_print:
        print(
            f"  ... omitted "
            f"{len(lora_module_names) - max_print} modules"
        )

    print("\n[Trainable LoRA parameters]")

    for name, param in trainable_lora[:max_print]:
        print(
            f"  TRAINABLE | {name} | "
            f"shape={tuple(param.shape)} | "
            f"dtype={param.dtype}"
        )

    print("\n[Frozen LoRA parameters]")

    if not frozen_lora:
        print("  None")
    else:
        for name, param in frozen_lora[:max_print]:
            print(
                f"  FROZEN | {name} | "
                f"shape={tuple(param.shape)}"
            )

    print("\n[Trainable non-LoRA parameters]")

    if not trainable_non_lora:
        print("  None")
    else:
        for name, param in trainable_non_lora[:max_print]:
            print(
                f"  TRAINABLE NON-LORA | {name} | "
                f"shape={tuple(param.shape)}"
            )

    # Count LoRA modules by broad branch.
    branch_counter = Counter()

    for name in lora_module_names:
        if ".audio_to_video_attn." in name:
            branch_counter["audio_to_video_attn"] += 1
        elif ".video_to_audio_attn." in name:
            branch_counter["video_to_audio_attn"] += 1
        elif ".audio_attn1." in name:
            branch_counter["audio_attn1"] += 1
        elif ".audio_attn2." in name:
            branch_counter["audio_attn2"] += 1
        elif ".attn1." in name:
            branch_counter["video_attn1"] += 1
        elif ".attn2." in name:
            branch_counter["video_attn2"] += 1
        else:
            branch_counter["other"] += 1

    print("\n[LoRA module counts by branch]")

    for branch, count in sorted(branch_counter.items()):
        print(f"  {branch}: {count}")

    print("=" * 100 + "\n")
    
_LORA_KEY_PATTERN = re.compile(
    r"^(?P<module>.+)\.lora_(?P<branch>A|B)"
    r"(?:\.[^.]+)?\.weight$"
)


def _normalize_ltx_lora_key(key: str) -> str:
    """
    Normalize common LTX/PEFT LoRA key formats.

    Examples:
        diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight

        base_model.model.transformer_blocks.0.attn1.to_q
        .lora_A.default.weight

    becomes:
        transformer_blocks.0.attn1.to_q.lora_A.weight
    """
    prefixes = (
        "diffusion_model.",
        "base_model.model.",
        "base_model.",
    )

    changed = True

    while changed:
        changed = False

        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True

    match = _LORA_KEY_PATTERN.match(key)

    if match is None:
        return key

    module_name = match.group("module")
    branch = match.group("branch")

    return f"{module_name}.lora_{branch}.weight"


def _inspect_motion_track_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[dict[str, Tensor], int, list[str]]:
    """
    Read the motion-track LoRA checkpoint and infer:
      1. normalized state dict
      2. LoRA rank
      3. exact target module names
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Motion-track LoRA checkpoint not found: "
            f"{checkpoint_path}"
        )

    raw_state_dict = load_file(
        checkpoint_path,
        device="cpu",
    )

    normalized_state_dict: dict[str, Tensor] = {}
    target_modules: set[str] = set()
    ranks: set[int] = set()
    skipped_keys: list[str] = []

    for raw_key, tensor in raw_state_dict.items():
        key = _normalize_ltx_lora_key(raw_key)
        match = _LORA_KEY_PATTERN.match(key)

        if match is None:
            skipped_keys.append(raw_key)
            continue

        module_name = match.group("module")
        branch = match.group("branch")

        normalized_state_dict[key] = tensor
        target_modules.add(module_name)

        if branch == "A":
            if tensor.ndim != 2:
                raise ValueError(
                    f"Unexpected LoRA-A shape:\n"
                    f"  key={raw_key}\n"
                    f"  shape={tuple(tensor.shape)}"
                )

            # PEFT Linear LoRA-A:
            # [rank, in_features]
            ranks.add(int(tensor.shape[0]))

    if not normalized_state_dict:
        raise RuntimeError(
            "No LoRA A/B tensors were found in motion-track "
            f"checkpoint: {checkpoint_path}"
        )

    if not ranks:
        raise RuntimeError(
            "No LoRA-A tensors were found, so rank cannot "
            "be inferred."
        )

    if len(ranks) != 1:
        raise RuntimeError(
            "The motion-track checkpoint contains multiple "
            f"LoRA ranks: {sorted(ranks)}. "
            "The current loader assumes a uniform rank."
        )

    rank = next(iter(ranks))
    sorted_targets = sorted(target_modules)

    logger.info(
        "Motion-track checkpoint inspection:\n"
        f"  checkpoint: {checkpoint_path}\n"
        f"  LoRA tensors: {len(normalized_state_dict)}\n"
        f"  target modules: {len(sorted_targets)}\n"
        f"  inferred rank: {rank}\n"
        f"  skipped non-LoRA tensors: {len(skipped_keys)}"
    )

    logger.info(
        "Motion-track LoRA target layers:\n"
        + "\n".join(
            f"  [{index:04d}] {name}"
            for index, name in enumerate(sorted_targets)
        )
    )

    if skipped_keys:
        logger.warning(
            "The following non-LoRA checkpoint tensors were "
            "not loaded:\n"
            + "\n".join(skipped_keys[:50])
        )

    return (
        normalized_state_dict,
        rank,
        sorted_targets,
    )
    
def _locate_named_lora_adapter(
    model: torch.nn.Module,
    adapter_name: str,
) -> list[str]:
    """
    Locate all PEFT LoRA layers containing the given adapter.
    """
    located_layers: list[str] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue

        adapter_found = False

        for container_name in (
            "lora_A",
            "lora_B",
            "lora_embedding_A",
            "lora_embedding_B",
            "lora_magnitude_vector",
        ):
            container = getattr(
                module,
                container_name,
                None,
            )

            if container is None:
                continue

            if adapter_name in container:
                adapter_found = True
                break

        if adapter_found:
            located_layers.append(module_name)

    return located_layers 
    


def initialize_fsdp_hierarchy(model) -> None:
    """
    Initialize the complete nested FSDP hierarchy from the outermost
    FSDP root before any nested FSDP module is called independently.

    This must run on every rank.
    """
    if not isinstance(model, FSDP):
        return

    is_root = model.check_is_root()

    if not is_root:
        raise RuntimeError(
            "The supplied model is not the outermost FSDP root. "
            "Please call initialize_fsdp_hierarchy() on the outer "
            "Transformer FSDP wrapper, not on spatial_track_encoder."
        )
