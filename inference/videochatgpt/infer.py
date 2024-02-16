"""Evaluation script for VideoChatGPT"""
import json
from pathlib import Path
from tqdm.auto import tqdm
from video_chatgpt import VideoChatGPT, DEFAULT_CAPTION_PROMPT, DEFAULT_SUMMARY_PROMPT

import logging
logging.basicConfig(level=logging.INFO, 
                    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

model = VideoChatGPT(
    model_name="/path/to/LLaVA-7B-Lightening-v1-1",
    projection_path="/path/to/video_chatgpt-7B.bin"
)
conv_mode = "video-chatgpt_v1"

video_dir = "/path/to/videos"
save_every = 100

captions = {}
summaries = {}
for ctr, vid_path in tqdm(enumerate(Path(video_dir).iterdir())):
    if not vid_path.suffix == '.mp4':
        continue
    try:
        with open(vid_path, "rb") as f:
            video = f.read()
        # caption
        caption = model.generate_text(video, prompt=DEFAULT_CAPTION_PROMPT)
        summary = model.generate_text(video, prompt=DEFAULT_SUMMARY_PROMPT)
    except:
        logging.error(f"FAILED TO PROCESS {vid_path}")
        continue
    captions[vid_path.stem] = caption
    summaries[vid_path.stem] = summary

    if ctr > 0 and ctr % save_every == 0:
        with open(f"video_chatgpt_{conv_mode}_captions.json", "w") as f:
            json.dump(captions, f)
        with open(f"video_chatgpt_{conv_mode}_summaries.json", "w") as f:
            json.dump(summaries, f)
with open(f"video_chatgpt_{conv_mode}_captions.json", "w") as f:
    json.dump(captions, f)
with open(f"video_chatgpt_{conv_mode}_summaries.json", "w") as f:
    json.dump(summaries, f)