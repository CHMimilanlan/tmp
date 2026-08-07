import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


# =========================
# 默认参数（可通过命令行覆盖）
# =========================
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 512
DEFAULT_NUM_FRAMES = 121
DEFAULT_FPS = 24

DEFAULT_OUTPUT_JSON = "left_to_right_tracks.json"
DEFAULT_OUTPUT_VIDEO = "left_to_right_track.mp4"


# 与 sparse_tracks.py 一致
_MIN_RADIUS = 2
_MAX_RADIUS = 8
_MAX_TRAIL = 50
_REF_SHORT_SIDE = 1080


def interpolate_spline(control_points: list[dict], num_samples: int) -> list[dict]:
    """
    对应 sparse_tracks.py 中 _interpolate_spline 的两点线性插值分支。
    两个点时，轨迹会匀速直线移动。
    """
    if len(control_points) != 2:
        raise ValueError("本示例只接受两个控制点。")

    if num_samples < 2:
        raise ValueError("num_samples 必须 >= 2。")

    a, b = control_points

    return [
        {
            "x": round(a["x"] + (b["x"] - a["x"]) * i / (num_samples - 1)),
            "y": round(a["y"] + (b["y"] - a["y"]) * i / (num_samples - 1)),
        }
        for i in range(num_samples)
    ]


def age_color_rgb(ratio: float) -> tuple[int, int, int]:
    """
    与 sparse_tracks.py 的 _age_color_batch 对应：
    old: blue -> green -> yellow -> new: red

    返回数值通道顺序为 RGB。
    """
    if ratio <= 1 / 3:
        tr = ratio * 3
        r, g, b = 0.0, tr, 1.0 - tr
    elif ratio <= 2 / 3:
        tr = (ratio - 1 / 3) * 3
        r, g, b = tr, 1.0, 0.0
    else:
        tr = (ratio - 2 / 3) * 3
        r, g, b = 1.0, 1.0 - tr, 0.0

    return (
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255)),
    )


def render_resolution(
    width: int,
    height: int,
    reference_short_side: int = _REF_SHORT_SIDE,
) -> tuple[int, int, float, float]:
    """
    对应 sparse_tracks.py 的 _render_resolution。
    在短边为 1080 的高分辨率画布绘制后，再缩小。
    """
    if height <= width:
        render_width = int(width * reference_short_side / height)
        render_height = reference_short_side
    else:
        render_width = reference_short_side
        render_height = int(height * reference_short_side / width)

    scale_x = render_width / width
    scale_y = render_height / height

    return render_width, render_height, scale_x, scale_y


def generate_track_frames(
    tracks: list[list[dict]],
    width: int,
    height: int,
):
    """
    按 sparse_tracks.py 的逻辑逐帧渲染轨迹。

    输出的数组已经执行官方的 RGB -> BGR 数值交换，
    应作为 RGB bytes 直接送入 FFmpeg。
    """
    render_width, render_height, scale_x, scale_y = render_resolution(
        width,
        height,
    )

    num_frames = max(len(track) for track in tracks)

    scaled_tracks = []
    for track in tracks:
        scaled_tracks.append(
            [
                {
                    "x": point["x"] * scale_x,
                    "y": point["y"] * scale_y,
                }
                for point in track
            ]
        )

    for frame_idx in range(num_frames):
        # 数值上按 RGB 绘制；OpenCV 不会自行转换通道，只是直接写数组。
        highres_rgb = np.zeros(
            (render_height, render_width, 3),
            dtype=np.uint8,
        )

        trail_start = max(0, frame_idx - _MAX_TRAIL)

        # 旧轨迹先画，新轨迹后画；新点会覆盖旧点。
        for track in scaled_tracks:
            end_idx = min(frame_idx, len(track) - 1)

            for point_idx in range(trail_start, end_idx + 1):
                point = track[point_idx]

                age = frame_idx - point_idx
                ratio = float(np.clip(1.0 - age / _MAX_TRAIL, 0.0, 1.0))

                radius = _MIN_RADIUS + (_MAX_RADIUS - _MIN_RADIUS) * ratio
                radius = max(1, int(round(radius)))

                x = int(round(point["x"]))
                y = int(round(point["y"]))

                if not (0 <= x < render_width and 0 <= y < render_height):
                    continue

                rgb = age_color_rgb(ratio)

                cv2.circle(
                    highres_rgb,
                    center=(x, y),
                    radius=radius,
                    color=rgb,
                    thickness=-1,
                    lineType=cv2.LINE_8,
                )

        # 与官方逻辑一致：高分辨率渲染后再双线性缩小。
        frame_rgb = cv2.resize(
            highres_rgb,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        # 官方 sparse_tracks.py 的关键一步：
        # RGB -> BGR，匹配 Motion Track IC-LoRA 训练格式。
        frame_for_ic_lora = frame_rgb[..., [2, 1, 0]].copy()

        yield frame_for_ic_lora


def save_h264_video(
    frames,
    output_path: str,
    width: int,
    height: int,
    fps: int,
):
    """
    保存为 H.264。

    使用 libx264rgb + CRF 0，尽量避免 yuv420p 对小彩色轨迹点的色度损失。
    输出依然是 H.264 MP4。
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未检测到 ffmpeg，请先安装 FFmpeg。")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",

        # 输入：Python 逐帧写入的三通道原始数据
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",

        "-an",

        # 输出：标准 H.264，兼容性更好
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "10",
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",

        str(output_path),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        for frame in frames:
            if frame.shape != (height, width, 3):
                raise ValueError(
                    f"帧尺寸错误：期望 {(height, width, 3)}，实际 {frame.shape}"
                )

            process.stdin.write(frame.tobytes())

        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()

    except BrokenPipeError:
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg 提前中断：\n{stderr}")

    if return_code != 0:
        raise RuntimeError(
            f"FFmpeg H.264 编码失败，返回码={return_code}\n{stderr}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成一条从左到右的水平轨迹视频（可自定义宽高）。",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="输出视频宽度")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="输出视频高度")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES, help="总帧数")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="帧率")

    # 起止点：未显式指定时，按宽高自动适配
    parser.add_argument("--start-x", type=int, default=None, help="起点 X（默认 width * 0.125）")
    parser.add_argument("--end-x", type=int, default=None, help="终点 X（默认 width * 0.875）")
    parser.add_argument("--y", type=int, default=None, help="轨迹 Y 坐标（默认 height / 2）")

    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON, help="输出 JSON 路径")
    parser.add_argument("--output-video", type=str, default=DEFAULT_OUTPUT_VIDEO, help="输出视频路径")
    return parser.parse_args()


def main():
    args = parse_args()

    width = args.width
    height = args.height
    num_frames = args.num_frames
    fps = args.fps

    if width <= 0 or height <= 0:
        raise ValueError(f"width/height 必须为正数，收到 width={width}, height={height}")

    start_x = args.start_x if args.start_x is not None else int(round(width * 0.125))
    end_x = args.end_x if args.end_x is not None else int(round(width * 0.875))
    y = args.y if args.y is not None else height // 2

    # 只给两个点，官方逻辑会产生一条水平、匀速、从左到右的轨迹。
    sparse_control_points = [
        {"x": start_x, "y": y},
        {"x": end_x, "y": y},
    ]

    left_to_right_track = interpolate_spline(
        sparse_control_points,
        num_frames,
    )

    # sparse_tracks.py 期望的格式：list[track]，即最外层仍需包一层。
    tracks = [left_to_right_track]

    Path(args.output_json).write_text(
        json.dumps(tracks, indent=2),
        encoding="utf-8",
    )

    save_h264_video(
        frames=generate_track_frames(
            tracks=tracks,
            width=width,
            height=height,
        ),
        output_path=args.output_video,
        width=width,
        height=height,
        fps=fps,
    )

    print(f"Saved JSON:  {args.output_json}")
    print(f"Saved H.264: {args.output_video}")
    print(f"Resolution: {width}x{height} @ {fps}fps")
    print(f"Start: ({start_x}, {y})")
    print(f"End:   ({end_x}, {y})")
    print(f"Frames: {num_frames}")


if __name__ == "__main__":
    main()
