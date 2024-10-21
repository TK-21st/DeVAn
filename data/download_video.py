"""
A script for downloading a video from YouTube.
"""
import shutil
import argparse
import yaml
import io
import os
import random
import tempfile
from typing import Union
from dataclasses import dataclass, field

from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import cleanup_description
from youtube_utils import read_vtt
from yt_dlp import YoutubeDL

@dataclass
class SentenceBertConfig:
    filter_with_sentencebert: bool = True
    sentence_bert_model: Union["SentenceTransformer", str] = 'sentence-transformers/all-MiniLM-L6-v2'
    similarity_threshold: float = 0.5

@dataclass
class LanguageFilterConfig:
    filter_with_spacy: bool = True
    vocab: str = "en_core_web_sm"
    threshold: float = 0.8

@dataclass
class YoutubeConfig:
    ids_fn: str
    video_dir: str
    out_dir: str
    thumbnail_width: int = 256
    verbose: bool = True
    redownload: bool = False
    shuffle: bool = False
    nofilter: bool = True
    skip_video: bool = True
    language_filter: LanguageFilterConfig = field(default_factory=lambda: LanguageFilterConfig())
    sentence_bert: SentenceBertConfig = field(default_factory=lambda: SentenceBertConfig())

    def __post_init__(self):
        if isinstance(self.language_filter, dict):
            self.language_filter = LanguageFilterConfig(**self.language_filter)
        if isinstance(self.sentence_bert, dict):
            self.sentence_bert = SentenceBertConfig(**self.sentence_bert)


def video_id_exists(video_id: str, video_dir: str) -> str:
    """ Check whether we've downloaded the video already."""
    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    if os.path.exists(video_path):
        return video_path
    return None

def download_id(video_id: str, cache_path: str, config: YoutubeConfig):
    """
    Download youtube
    """
    ydl_opts = {
        'writedescription': False,
        'writeinfojson': False,
        'write_all_thumbnails': False,
        'writeautomaticsub': True,
        'writesubtitles': True,
        'subtitlesformat': 'vtt',
        'format': "bv*[height=360][ext=mp4]+ba[height=360][ext=m4a]/b[height=360][ext=mp4] / bv*+ba/b",
        'ignoreerrors': True,
        'skip_download': True,
        'subtitleslangs': ['en','EN','eng','ENG','en-gb','en-us','en-GB','en-US','EN-GB','EN-US','english','English','ENGLISH'],
        'no_warnings': True,
        "outtmpl": f"{cache_path}/%(id)s/%(id)s.%(ext)s",
        "writethumbnail": True
    }

    video_fn = video_id_exists(video_id, config.video_dir)
    should_download_video = not config.skip_video and (
        (video_fn is None) or (video_fn is not None and config.redownload)
    )
    ydl_opts['skip_download'] = not should_download_video

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.sanitize_info(ydl.extract_info(video_id, download=True))
        info_dir = os.path.join(cache_path, video_id)

    info.pop('thumbnail', None)
    info.pop('playlist', None)
    info.pop('is_live', None)
    info.pop('subtitles', None)
    info.pop('protocol', None)
    info.pop('playlist_index', None)
    info.pop('extractor_key', None)
    info.pop('extractor', None)
    info.pop('url', None)
    info.pop('automatic_captions', None)
    info.pop('formats', None)
    info.pop('http_headers', None)

    if info['description'] is not None:
        info['description'] = cleanup_description(
            info['description'],
            title=info['title'],
            filter_with_sentencebert=config.sentence_bert.filter_with_sentencebert,
            sentence_bert_model=config.sentence_bert.sentence_bert_model,
            similarity_threshold=config.sentence_bert.similarity_threshold
        )

    # Lang detect
    if config.language_filter.filter_with_spacy:
        text = info['title']
        if info['description'] is not None:
            text += info['description']

        res = NLP(text)
        info['_lang_prob'] = _prob = res._.language_score
        info['_lang'] = _lang = str(res._.language)
        if (_lang != 'en') or (_prob < 0.8):
            info['_failreason'] = 'maybe not english'
            if config.verbose:
                print(f"Skipping bc langdetect: lang is {_lang} p={_prob:.3f}", flush=True)
            return info

    info['video_fn'] = video_fn
    if should_download_video:
        video_fn = video_id_exists(video_id, info_dir)
        if video_fn is not None:
            shutil.copy(video_fn, config.video_dir)
            print(f"Copied video {video_id} from {video_fn} to {config.video_dir}", flush=True)
        info['video_fn'] = video_fn

    subtitle_filepath = os.path.join(cache_path, video_id, f"{video_id}.en.vtt")
    transcript = {
        "asr": "",
        "start_time_list": [],
        "end_time_list": [],
        "phrase_list": []
    }
    if os.path.exists(subtitle_filepath):
        try:
            transcript_raw = read_vtt(subtitle_filepath)
            if isinstance(transcript_raw, list):
                words = [_[0] for _ in transcript_raw]
                transcript = {
                    "asr": " ".join(words),
                    "start_time_list": [int(_[1]*1e3) for _ in transcript_raw],
                    "end_time_list": [int(_[2]*1e3) for _ in transcript_raw],
                    "phrase_list": words
                }
        except Exception as e:
            print(f"Failed to read subtile at {subtitle_filepath}: {e}")

    info['subtitles'] = transcript

    # last but not least get thumbnails
    thumbnail_path = [_ for _ in info['thumbnails'] if 'filepath' in _]
    if thumbnail_path:
        info['image'] = process_thumbnail(thumbnail_path[0]['filepath'])
    else:
        info['image'] = None

    info['category'] = info.pop("categories", [""])[0]
    info.pop('thumbnails', None)
    return info

def PIL_to_bytes(image: Image) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def process_thumbnail(path: str, width=256) -> bytes:
    with open(path, "rb") as f:
        img = Image.open(f).convert("RGB")
        wpercent = (width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((width, hsize), Image.Resampling.LANCZOS)
        return PIL_to_bytes(img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download")
    parser.add_argument(
        '--config',
        type=str,
        help='Config file (yaml).',
        required=True
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = YoutubeConfig(**yaml.safe_load(f))

    # check paths
    assert os.path.exists(config.ids_fn)
    video_dir = config.video_dir
    out_dir = config.out_dir
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)


    if config.language_filter.filter_with_spacy is True:
        import spacy
        import spacy_fastlang
        NLP = spacy.load(config.language_filter.vocab)
        NLP.add_pipe("language_detector")

    if config.sentence_bert.filter_with_sentencebert is True:
        from sentence_transformers import SentenceTransformer
        if isinstance(config.sentence_bert.sentence_bert_model, str):
            config.sentence_bert.sentence_bert_model = SentenceTransformer(
                config.sentence_bert.sentence_bert_model
            )

    df = pd.read_json(str(config.ids_fn), lines=True, compression="gzip")
    unique_video_ids = list(set(df['meta'].apply(lambda m: m['yt_id']).values.tolist()))

    if config.shuffle:
        random.shuffle(unique_video_ids)

    results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        pbar = tqdm(total=len(unique_video_ids))
        for video_id in unique_video_ids:
            pbar.set_description(video_id)
            try:
                info = download_id(video_id, cache_path=tmp_dir, config=config)
                pbar.update()
                info['youtube_id'] = video_id
                results.append(info)
            except Exception as e:
                print(f"Failed to download {video_id}: {e}")
                pbar.update()
                continue
        pbar.close()

    pd.DataFrame(results).to_parquet(os.path.join(config.out_dir, f"{Path(config.ids_fn).stem}.snappy.parquet"))