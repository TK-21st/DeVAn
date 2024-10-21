import os
from pathlib import Path
import argparse
from download_video import YoutubeConfig, download_id
import pandas as pd
import random
import yaml
import tempfile
import concurrent.futures
from tqdm.auto import tqdm
from dataclasses import dataclass

@dataclass
class YoutubeConfigMultiProcess(YoutubeConfig):
    max_workers: int = -1

def process_video(video_id: str, tmp_dir: str, config: YoutubeConfig):
    """
    Process a single video
    """
    try:
        info = download_id(video_id, cache_path=tmp_dir, config=config)
        info['youtube_id'] = video_id
        return info
    except Exception as e:
        print(f"Failed to download {video_id}: {e}")
        return None

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
        config = YoutubeConfigMultiProcess(**yaml.safe_load(f))

    if config.language_filter.filter_with_spacy:
        import spacy
        import spacy_fastlang
        NLP = spacy.load(config.language_filter.vocab)
        NLP.add_pipe("language_detector")

    if config.sentence_bert.filter_with_sentencebert:
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
        # Use concurrent.futures to process videos concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Submit all tasks
            future_to_video = {executor.submit(process_video, video_id, tmp_dir, config): video_id for video_id in unique_video_ids}

            # Process as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_video), total=len(unique_video_ids), desc="Processing videos", dynamic_ncols=True):
                video_id = future_to_video[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as exc:
                    print(f'{video_id} generated an exception: {exc}')

    pd.DataFrame(results).to_parquet(os.path.join(config.out_dir, f"{Path(config.ids_fn).stem}.snappy.parquet"))