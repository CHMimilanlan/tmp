#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Configurable paths / parameters
# ============================================================
DISTILLED_CHECKPOINT_PATH="/data/vjuicefs_ai_camera_jgroup_video/public_data/Video_Data/VideoAudioData/VideoAudioModels/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
SPATIAL_UPSAMPLER_PATH="/data/vjuicefs_ai_camera_jgroup_video/public_data/Video_Data/VideoAudioData/VideoAudioModels/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT="/data/vjuicefs_ai_camera_jgroup_video/public_data/Video_Data/VideoAudioData/VideoAudioModels/gemma-3-12b-it-qat-q4_0-unquantized"
LORA_SCALE="1.0"
BATCH_JSON="/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11194554/WorkSpace/ProcessSpatialAudio/GenerateEvaluationVideo/results/samples.json"

CHECKPOINT_DIR="/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11194554/WorkSpace/LTX-2/packages/ltx-trainer/outputs/av2av_ic_lora_full_train_monodiff_simple_summary_v1/checkpoints"
LORA_PATH="${CHECKPOINT_DIR}/lora_weights_step_05250.safetensors"
SIMPLE_SUMMARY_WEIGHTS="${CHECKPOINT_DIR}/simple_summary_step_05250.safetensors"
OUTPUT_PATH="evaluation_results/"
CUDA_DEVICE="3"

# Optional track modules can be enabled by adding either argument below:
#   --track-prope-weights /path/to/track_prope_step_05250.safetensors
#   --spatial-track-encoder-weights /path/to/spatial_track_encoder_step_05250.safetensors

# ============================================================
# Run
# ============================================================
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python -m ltx_pipelines.ic_lora_customize_batch \
  --distilled-checkpoint-path "${DISTILLED_CHECKPOINT_PATH}" \
  --spatial-upsampler-path "${SPATIAL_UPSAMPLER_PATH}" \
  --gemma-root "${GEMMA_ROOT}" \
  --lora "${LORA_PATH}" "${LORA_SCALE}" \
  --simple-summary-weights "${SIMPLE_SUMMARY_WEIGHTS}" \
  --batch-json "${BATCH_JSON}" \
  --output-path "${OUTPUT_PATH}" \
  --height 1024 \
  --width 1536 \
  --num-frames 121 \
  --frame-rate 24 \
  --seed 42 \
  --skip-stage-2 \
  --decode-mode
