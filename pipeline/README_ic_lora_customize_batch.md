# IC-LoRA 批量推理与轨迹模块使用说明

本文说明如何运行 `pipeline/ic_lora_customize_batch.py`，以及如何选择性加载
Spatial Track Encoder 和 Track-PRoPE。该脚本基于 LTX-2.3 的两阶段 distilled
IC-LoRA 推理流程。

## 1. 运行前提

请先准备可正常运行的 LTX-2.3 环境，并确认 Python 能导入：

- `ltx_core`
- `ltx_pipelines`
- `torch`
- `safetensors`（加载 `.safetensors` 权重时需要）

本项目的 `pipeline/transformer.py` 包含 Track-PRoPE 修改；实际安装到环境中的
`ltx_core` 也必须包含相同修改。此外，`ModalitySpec` 必须定义 `track_xy` 和
`track_valid` 字段，否则脚本会主动报错，而不是静默忽略轨迹。

首先查看当前 LTX-2.3 版本提供的完整基础参数：

```bash
python pipeline/ic_lora_customize_batch.py --help
```

下面示例中的 checkpoint 参数名来自 LTX-2.3 的
`default_2_stage_distilled_arg_parser`。若安装版本的 `--help` 输出不同，应以实际
输出为准。

## 2. 所需文件

运行完整的批量 Track-PRoPE 推理通常需要：

1. LTX-2.3 distilled checkpoint。
2. LTX-2 spatial upsampler 权重。
3. Gemma 文本编码器目录。
4. IC-LoRA 权重。
5. 每个样例的 reference video。
6. 每个样例的 track `.pt` 文件。
7. 可选的 Spatial Track Encoder 权重。
8. 可选的 Track-PRoPE 权重。

建议目录结构：

```text
project/
├── pipeline/
│   └── ic_lora_customize_batch.py
├── models/
│   ├── ltx-2.3-distilled.safetensors
│   ├── spatial-upscaler.safetensors
│   ├── gemma/
│   ├── ic_lora.safetensors
│   ├── spatial_track_encoder_step_01000.safetensors
│   └── track_prope_step_01000.safetensors
└── test_data/
    ├── batch.json
    ├── videos/
    ├── tracks/
    └── outputs/
```

## 3. Track 文件格式

Track 文件使用 `torch.save` 保存，支持以下两种格式。

### 3.1 直接保存坐标 Tensor

```python
import torch

# shape: [T, 2]，最后一维依次为 x、y
track_xy = torch.tensor([
    [0.10, 0.25],
    [0.12, 0.27],
    [0.15, 0.30],
], dtype=torch.float32)

torch.save(track_xy, "sample_000.pt")
```

此格式默认所有时间点均有效。

### 3.2 保存坐标和有效性 Mask

```python
import torch

torch.save(
    {
        "track_xy": torch.tensor(
            [[0.10, 0.25], [0.12, 0.27], [0.00, 0.00]],
            dtype=torch.float32,
        ),                         # [T, 2]
        "track_valid": torch.tensor([True, True, False]),  # [T]
    },
    "sample_001.pt",
)
```

脚本会将非有限坐标标为无效，并把轨迹线性插值到目标音频 token 数；有效性 Mask
使用 nearest 插值。一个 `.pt` 文件只能表示一个测试样例，但允许旧格式中的
singleton batch 维度 `[1, T, 2]`。

## 4. Batch JSON 格式

推荐使用带 `samples` 字段的对象：

```json
{
  "samples": [
    {
      "prompt": "A person walks from the left side of the room to the right.",
      "reference_video": "videos/reference_000.mp4",
      "track": "tracks/sample_000.pt",
      "reference_strength": 1.0,
      "seed": 42,
      "height": 768,
      "width": 1152,
      "num_frames": 121,
      "frame_rate": 24.0,
      "output_path": "outputs/sample_000.mp4"
    },
    {
      "prompt": "A dog runs toward the camera in a park.",
      "reference_video": "videos/reference_001.mp4",
      "track": "tracks/sample_001.pt",
      "reference_strength": 0.8,
      "seed": 123,
      "output_path": "outputs/sample_001.mp4"
    }
  ]
}
```

也可以把顶层直接写成数组。字段说明如下：

| 字段 | 必需 | 说明 |
| --- | --- | --- |
| `prompt` | 是 | 文本提示词。 |
| `reference_video` | 是 | IC-LoRA reference video。 |
| `track` | 是 | 当前样例的 `.pt` 轨迹文件。 |
| `reference_strength` | 否 | Reference conditioning 强度，默认 `1.0`。 |
| `seed` | 否 | 覆盖命令行的全局 seed。 |
| `height` / `width` | 否 | 覆盖命令行输出分辨率；必须满足 LTX 两阶段分辨率约束。 |
| `num_frames` | 否 | 覆盖命令行帧数。 |
| `frame_rate` | 否 | 覆盖命令行帧率。 |
| `output_path` | 否 | 当前样例的输出文件名；其中的目录部分会被忽略。 |

`reference_video` 和 `track` 的相对路径均相对于 JSON 文件所在目录，而不是
当前 shell 的工作目录。`output_path` 只用于指定输出文件名；所有结果都会写入
命令行 `--output-path` 指定的父目录之下。如果未提供该字段，文件名依次为
`sample_0000.mp4`、`sample_0001.mp4`。

### 4.1 输出子目录命名规则

命令行 `--output-path` **必须是父文件夹地址**。脚本根据所加载的额外模块，在
该父文件夹下自动创建一层结果目录：

| 模块权重参数 | 自动创建的子目录 |
| --- | --- |
| 两个权重都不传 | `vanilla_results/` |
| 只传 `--track-prope-weights /path/track_prope_step_01000.safetensors` | `track_prope_step_01000.safetensors/` |
| 只传 `--spatial-track-encoder-weights /path/spatial_step_01000.safetensors` | `spatial_step_01000.safetensors/` |
| 两个都传 | `track_prope_step_01000.safetensors__spatial_step_01000.safetensors/` |

目录名使用权重文件的完整 filename，包含 `.safetensors`、`.pt` 等扩展名。
两个权重都存在时，顺序固定为 Track-PRoPE、Spatial Track Encoder，并使用
`__` 连接。例如 `--output-path test_data/outputs` 最终可能产生：

```text
test_data/outputs/
└── track_prope_step_01000.safetensors__spatial_step_01000.safetensors/
    ├── sample_0000.mp4
    └── sample_0001.mp4
```

## 5. 批量推理

### 5.1 同时使用 Spatial Track Encoder 和 Track-PRoPE

```bash
python pipeline/ic_lora_customize_batch.py \
  --distilled-checkpoint-path models/ltx-2.3-distilled.safetensors \
  --spatial-upsampler-path models/spatial-upscaler.safetensors \
  --gemma-root models/gemma \
  --lora models/ic_lora.safetensors 1.0 \
  --spatial-track-encoder-weights \
    models/spatial_track_encoder_step_01000.safetensors \
  --track-prope-weights models/track_prope_step_01000.safetensors \
  --batch-json test_data/batch.json \
  --height 768 \
  --width 1152 \
  --num-frames 121 \
  --frame-rate 24 \
  --seed 42 \
  --output-path test_data/outputs
```

命令行中的尺寸、帧数、帧率和 seed 是默认值；JSON 中的同名字段优先。

### 5.2 只启用 Track-PRoPE

省略 `--spatial-track-encoder-weights`：

```bash
python pipeline/ic_lora_customize_batch.py \
  --distilled-checkpoint-path models/ltx-2.3-distilled.safetensors \
  --spatial-upsampler-path models/spatial-upscaler.safetensors \
  --gemma-root models/gemma \
  --lora models/ic_lora.safetensors 1.0 \
  --track-prope-weights models/track_prope_step_01000.safetensors \
  --batch-json test_data/batch.json \
  --output-path test_data/outputs
```

启用 Track-PRoPE 后，每个样例都必须提供有效的 `track`。

### 5.3 只加载 Spatial Track Encoder

省略 `--track-prope-weights`：

```bash
python pipeline/ic_lora_customize_batch.py \
  --distilled-checkpoint-path models/ltx-2.3-distilled.safetensors \
  --spatial-upsampler-path models/spatial-upscaler.safetensors \
  --gemma-root models/gemma \
  --lora models/ic_lora.safetensors 1.0 \
  --spatial-track-encoder-weights \
    models/spatial_track_encoder_step_01000.safetensors \
  --batch-json test_data/batch.json \
  --output-path test_data/outputs
```

### 5.4 显存不足时跳过第二阶段

追加：

```bash
--skip-stage-2
```

输出分辨率将变为目标高度和宽度的一半，并跳过空间放大及第二阶段 refinement。

## 6. 不使用额外轨迹模块的单样例模式

若不传两个模块权重参数，可以继续使用原有的单样例方式。单样例模式不接受
Track-PRoPE，因为 `track` 是从 batch JSON 中逐样例读取的。

```bash
python pipeline/ic_lora_customize_batch.py \
  --distilled-checkpoint-path models/ltx-2.3-distilled.safetensors \
  --spatial-upsampler-path models/spatial-upscaler.safetensors \
  --gemma-root models/gemma \
  --lora models/ic_lora.safetensors 1.0 \
  --prompt "A cinematic shot of a person walking across a room." \
  --video-conditioning test_data/videos/reference_000.mp4 1.0 \
  --height 768 \
  --width 1152 \
  --num-frames 121 \
  --frame-rate 24 \
  --seed 42 \
  --output-path test_data/outputs
```

## 7. 可选 conditioning attention mask

可以使用灰度 mask video 控制 reference conditioning 的空间强度：

```bash
--conditioning-attention-mask test_data/masks/mask.mp4 0.5
```

Mask 像素值会归一化到 `[0, 1]`，然后与命令中的强度相乘。`0.0` 表示忽略
IC-LoRA reference conditioning，`1.0` 表示完整强度。

## 8. 常见错误

### `ModalitySpec does not expose track_xy/track_valid`

当前 Python 环境使用了未经 Track-PRoPE 修改的 LTX core。请把本项目相应修改
合并到实际安装的 `ltx_core`，并确认 Python 导入的不是另一个环境中的旧版本。

### `No track_prope parameters ... match the loaded transformer`

指定文件不是 Track-PRoPE checkpoint，或者 checkpoint 的参数名与
`audio_track_prope` 模块不匹配。请使用 `trainer.py` 保存的 Track-PRoPE 权重。

### `No spatial_track_encoder parameters ... match the loaded transformer`

指定文件中没有 `spatial_track_encoder` 参数，或 encoder 配置与训练时不一致。
当前推理配置必须与训练配置保持一致：`dim=128`、`video_t=16`、`video_h=8`、
`video_w=12`、`audio_t=122`、`num_heads=8`。

### `--track-prope-weights requires a track for every sample`

启用了 Track-PRoPE，但某个 JSON 样例没有有效的 `track`。请为每条样例填写
轨迹路径并检查文件是否存在。

### 输出路径与预期不同

请确认 `--output-path` 传入的是父文件夹而不是 `.mp4` 文件。脚本一定会先根据
模块权重创建 `vanilla_results` 或权重名称子目录，再把 JSON 中 `output_path`
的文件名（或自动生成的 `sample_XXXX.mp4`）写入该目录。
