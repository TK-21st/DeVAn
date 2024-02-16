"""Evaluation Script for ImageBind-LLM

Please follow setup instructions in https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/imagebind_LLM before running the following sript.
Please this script in your LLaMA-Adapter/imagebind_LLM root directory.
"""
import ImageBind.data as data
import llama
import os
from pathlib import Path
from tqdm.auto import tqdm
import json

llama_dir = "/path/to/your/image_bind_ckpt"

# checkpoint will be automatically downloaded
model = llama.load("7B", llama_dir, knn=True)
model.eval()

videos = [p for p in Path("/path/to/all/videos").iterdir() if p.suffix == ".mp4"]

output_path = "/your/outputt/path"
with open(os.path.join(output_path, f"imagebind_llm_7b_captions.jsonl"), "w") as f_caption, open(os.path.join(output_path, f"imagebind_llm_7b_summaries.jsonl"), "w") as f_summary:
    for file_path in tqdm(videos):
        try:
            video = data.load_and_transform_video_data([str(file_path)], device='cuda')
        except Exception as e:
            print(f"Failed for {file_path}, skipped: {e}")
            continue
        video_weight = 1
        inputs = {'Video': [video, video_weight]}

        caption = model.generate(
            inputs,
            [llama.format_prompt("Summarize this video in ONE sentence.")],
            max_gen_len=256
        )[0]
        summary = model.generate(
            inputs,
            [llama.format_prompt("Describe this video in three to ten sentences.")],
            max_gen_len=256
        )[0]
        f_caption.write(
            json.dumps({
                "video_id":file_path.stem, # this assumes that the video path is video_id
                "caption": caption
            }) + "\n"
        )
        f_summary.write(
            json.dumps({
                "video_id":file_path.stem,
                "summary":summary
            }) + "\n"
        )