# DeVAn Evaluation Dataset

## Download Video
use `download_video.py` as main entry script for downloading videos.
```bash
python download_video_multiprocessing.py --config /your/config/file # refer to download_youtube_config_multiprocessing.yaml for reference
python postprocess_video.py --config /your/config/file # this will split the full downloaded videos into chunks
```

In the configuration file, these configurations are the most relevant:
```
ids_fn: /path/to/video_ids # the jsonl file that contains the data
video_dir: /output/video/file/path # the directory into which the full mp4 videos are downloaded from yt
out_dir: /output/video/final/path # the directory into which the split mp4 videos and all metadata from yt are stored.
```

## Setup
```bash
conda create -n devan python=3.10 -y
conda activate devan
conda install conda-forge::ffmpeg
pip install beautifulsoup4 PyYAML yt-dlp pillow numpy pandas tqdm lxml pyarrow
```


## Additional Requirement for custom dataset
The script provided supports filtering videos based on langauge and content of the metadata and uses a combinatino of `spacy` and `SentenceTransformer`.
You don't need to do this for the provided videos since filtering has already been done.

To use these additional filtering, the following packages are required:
```
sentence_transformers scikit-learn spacy spacy_fastlang
```

Refer to `download_youtube_config_with_filter.yaml` for example config file.