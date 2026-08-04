import os
import argparse
import torch
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
from typing import List
from pathlib import Path
from tqdm import tqdm

DEFAULT_VIDEO_NAME = "spatialized_video.mp4"

DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

DEFAULT_CAPTION_INSTRUCTION = """\
Analyze this audio-visual media and write a generation-friendly caption in the following EXACT format. Fill in ALL sections exactly once.

[VISUAL]: <Describe the visible scene in temporal order: main subjects, objects, setting, camera view, actions, movements, colors, lighting, and important changes over time. Focus on details useful for recreating the video. Maximum 120 words.>

[SPEECH]: <Describe spoken content only if clear. If there is repeated chanting, singing, or repeated words, summarize it instead of listing every repetition. If speech is unclear, write "[unclear speech]". If no clear speech, write "None". Maximum 40 words.>

[SOUNDS]: <Describe non-speech audio: music, ambient sounds, object sounds, sound effects, volume changes, rhythm, and which visible object or action likely produces each sound. Maximum 80 words.>

[TEXT]: <Transcribe only important, clearly readable, unique on-screen text. List each unique text only once. If the same text appears repeatedly across frames or signs, write it once and mention that it repeats. Ignore tiny, blurry, partially visible, or irrelevant background text. Maximum 5 text items. If no important readable text, write "None".>

Rules:
- Output exactly four sections: [VISUAL], [SPEECH], [SOUNDS], [TEXT].
- Each section heading must appear exactly once.
- Do not add [MOTION], [AUDIO], [SUMMARY], JSON, bullets outside [TEXT], or any extra headings.
- Do not repeat the same word, phrase, line, or visible text more than 2 times.
- For repeated speech, repeated signs, repeated subtitles, or repeated OCR text, summarize the repetition instead of expanding it.
- Describe only what is visible or audible; do not invent unseen causes.
"""


def MakeCaption(video_path, audio_path, caption_content, processor, sampling_params, llm):
    user_content = []
    user_content.append({"type": "video", "video": str(video_path)})
    user_content.append({"type": "audio", "audio": str(audio_path)})
    user_content.append({"type": "text", "text": caption_content})

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    use_audio_in_video = False
    
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    inputs = {
        'prompt': text,
        'multi_modal_data': {},
        "mm_processor_kwargs": {
            "use_audio_in_video": use_audio_in_video,
        },
    }

    if images is not None:
        inputs['multi_modal_data']['image'] = images
    if videos is not None:
        inputs['multi_modal_data']['video'] = videos
    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios

    outputs = llm.generate([inputs], sampling_params=sampling_params)

    print("done!!")
    response_content = outputs[0].outputs[0].text
    print(response_content)
    return response_content

def read_txt_to_list(txt_path: str, skip_empty: bool = True) -> List[str]:
    """
    读取 txt 文件，每一行作为 list 中的一个元素。

    Args:
        txt_path: txt 文件路径
        skip_empty: 是否跳过空行，默认跳过

    Returns:
        List[str]: txt 中每一行组成的列表
    """
    data: List[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if skip_empty and not line:
                continue
            data.append(line)
            
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Generate video-audio captions using Qwen3-Omni.")
    parser.add_argument(
        "--txt_path_list",
        type=str,
        nargs="+",
        required=True,
        help="One or more txt file paths. Each txt contains a list of mp4 paths (one per line).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/vjuicefs_ai_camera_jgroup_video/public_data/Video_Data/VideoAudioData/VideoAudioModels/Qwen3-Omni-30B-A3B-Captioner",
        help="Path to the Qwen3-Omni model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    MODEL_PATH = args.model_path
    # MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    txt_path_list = args.txt_path_list

    inputs = []
    for txt_path in txt_path_list:
        inputs += read_txt_to_list(txt_path)

    llm = LLM(
            model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.88,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={'image': 3, 'video': 3, 'audio': 3},
            max_num_seqs=8,
            max_model_len=32768,
            seed=1234,
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    # 失败样本日志路径：与第一个输入 txt 同目录，便于查看
    first_txt_dir = Path(txt_path_list[0]).resolve().parent
    failed_log_path = first_txt_dir / "failed_samples.txt"
    print(f"[Info] Failed samples will be logged to: {failed_log_path}")

    for sample in tqdm(inputs):
        try:
            # breakpoint()
            sample_dir = Path(sample).parent
            video_path = sample_dir / DEFAULT_VIDEO_NAME
            audio_path = sample_dir / "source_mono.wav"
            caption_content = DEFAULT_CAPTION_INSTRUCTION
            caption_content = MakeCaption(video_path, audio_path, caption_content, processor, sampling_params, llm)

            # 保存 caption_content 到 txt 文件
            caption_txt_path = sample_dir / "video_audio_caption.txt"
            try:
                with open(caption_txt_path, "w", encoding="utf-8") as f:
                    f.write(caption_content)
                print(f"[Saved] {caption_txt_path}")
            except Exception as e:
                print(f"[Error] Failed to save caption to {caption_txt_path}: {e}")
                raise
        except Exception as e:
            err_msg = f"{sample}\t{type(e).__name__}: {e}"
            print(f"[Failed] {err_msg}")
            try:
                with open(failed_log_path, "a", encoding="utf-8") as fout:
                    fout.write(err_msg + "\n")
                    fout.flush()
                    try:
                        os.fsync(fout.fileno())
                    except Exception:
                        pass
            except Exception as log_e:
                print(f"[Error] Failed to write failed log: {log_e}")
            continue


if __name__ == '__main__':
    main()
