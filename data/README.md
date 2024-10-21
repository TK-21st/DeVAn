# DeVAn Evaluation Dataset

## Download Video
use `download_video.py` as main entry script for downloading videos.
```bash
python download_video.py --config /your/config/file # refer to download_youtube_config.yaml for reference
```

## Setup
If
```bash
conda create -n devan python=3.10 -y
conda activate devan
pip install beautifulsoup4 PyYAML yt-dlp pillow numpy pandas tqdm lxml
```

Requirement
```
beautifulsoup4 PyYAML yt-dlp pillow numpy pandas tqdm
```

##
Additional Requirements
```
sentence_transformers scikit-learn spacy spacy_fastlang
```