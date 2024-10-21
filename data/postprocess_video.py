import os
import shutil
import glob
import argparse
import pandas as pd
import subprocess
import concurrent.futures
from tqdm.auto import tqdm
from typing import List, Tuple
import yaml

def split_video(params: Tuple[str, str, str, str]) -> Tuple[str, bool]:
    input_file, output_file, start_time, end_time = params
    if float(start_time) == 0 and float(end_time) == -1: # (0,-1) indicates full video without splitting
        try:
            shutil.copy(input_file, output_file)
            return (output_file, True, None)
        except Exception as e:
            return (output_file, False, str(e))

    command = [
        "ffmpeg",
        "-y", # overwrite
        "-i", str(input_file),
        "-ss", str(start_time),
        "-to", str(end_time),
        "-c", "copy",
        output_file
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return (output_file, True, None)
    except subprocess.CalledProcessError as e:
        return (output_file, False, str(e))

def process_videos(video_tasks: List[Tuple[str, str, str, str]], max_workers: int = 4) -> None:
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(split_video, task) for task in video_tasks]

        with tqdm(total=len(video_tasks), desc="Processing videos",) as pbar:
            for future in concurrent.futures.as_completed(futures):
                output_file, success, exc = future.result()
                if success:
                    pbar.set_postfix_str(f"Created: {output_file}", refresh=True)
                else:
                    pbar.set_postfix_str(f"Failed: {output_file}", refresh=True)
                    print(exc)
                pbar.update()

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
        config = yaml.safe_load(f)

    # check exists
    assert os.path.exists(config['ids_fn'])
    video_dir = config['video_dir']
    out_dir = config['out_dir']
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)


    df = pd.read_json(config['ids_fn'], lines=True, compression="gzip")
    video_fns = {os.path.basename(path):path for path in glob.glob(os.path.join(config['video_dir'], "*.mp4"))}
    df.loc[:, "video_fns"] = df['meta'].apply(lambda m: video_fns.get(m['yt_id'] + ".mp4", pd.NA))
    print(f"Total annotations: {len(df)}, videos found {(~df['video_fns'].isna()).sum()}")
    df = df[~df['video_fns'].isna()]

    def prepare_dataframe(row):
        input_fn, start_time, end_time, devan_id = row['video_fns'], row['meta']['start'], row['meta']['end'], row['id']
        output_fn = os.path.join(config['out_dir'], devan_id + ".mp4")
        return (input_fn, output_fn, start_time, end_time)

    video_tasks = df.apply(prepare_dataframe, axis=1).tolist()
    process_videos(video_tasks, max_workers=config['max_workers'])