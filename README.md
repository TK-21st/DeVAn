# DeVAn: Dense Video Annotation for Video-Language Models
This repository contains code and data related to submission to ACL ARR 2024 - _DeVAn: Dense Video Annotation for Video-Language Models_.

For more details on the dataset and example videos and annotations, refer to our [website](https://anonymous.4open.science/w/DeVAn/).

## Data
Please see `data/devan_acl24_release_v1.jsonl.gz` for V1 of release data. 
V1 release contains 8170 videos, their corresponding youtube id, start/end timestamp, ground truth captions/summaries and predicted caption/summaries from models evaluated.

## Evaluation

### Inference
To reproduce inference results reported in the manuscript, please first ensure that you have downloaded the video data following steps listed above.

1. Video-ChatGPT: Please follow descriptions in [official repo](https://github.com/mbzuai-oryx/Video-ChatGPT), then modify `inference/videochatgpt/infer.py` based on your input/output/ckpt paths.
2. ImageBind-LLM: Please follow descriptions in [official repo](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/imagebind_LLM), then modify `inference/infer_imagebind_llm.py` based on your input/output/ckpt paths.
3. Video-LLaMA: Please follow descriptions in [official repo](https://github.com/DAMO-NLP-SG/Video-LLaMA), then modify `inference/infer_videollama.py` based on your input/output/ckpt paths.
4. VideoCoCa: Code and ckpt for our finetuned VideoCoCa model will be released upon publication.

### Evaluation
To reproduce the evaluation results reported in the manuscript, please first ensure that you have completed the inference step above.

For generation tasks, refer to `evaluation/ngram_metrics.py` and `evaluation/model_based_metrics.py` to compute N-gram based and model-based metrics for model inference results.


## Leaderboard
We evaluate a range of different models on both video-to-text generation and text-to-video retrieval tasks.

### Caption Tasks
|           Model           | Audio | BLEU-4 | ROUGE-L | CIDEr | BLEURT | R@1 | R@5 | R@10 |
|:-------------------------:|:-----:|:------:|:-------:|:-----:|:------:|:---:|:---:|:----:|
|        Human (Avg)        |  Raw  |   6.3  |   32.1  |  53.9 |  50.5  |  -  |  -  |   -  |
|        Human (Min)        |  Raw  |   4.5  |   29.5  |  47.1 |  48.6  |  -  |  -  |   -  |
|       ImageBind-LLM       |  N/A  |   0.3  |   20.0  |  2.1  |  34.0  |  -  |  -  |   -  |
| Video-LLaMA2-Instruct 13B |  N/A  |   0.1  |   7.9   |  0.0  |  47.2  |  -  |  -  |   -  |
| Video-LLaMA2-Instruct 13B |  Raw  |   0.1  |   7.9   |  0.0  |  47.1  |  -  |  -  |   -  |
|  Video-LLaMA2-Instruct 7B |  N/A  |   0.1  |   10.8  |  0.0  |  43.6  |  -  |  -  |   -  |
|  Video-LLaMA2-Instruct 7B |  Raw  |   0.1  |   10.8  |  0.0  |  43.6  |  -  |  -  |   -  |
|        VideoChatGPT       |  N/A  |   0.4  |   19.9  |  2.0  |  40.5  |  -  |  -  |   -  |
|         VideoCoCa         |  N/A  |   0.2  |   13.2  |  2.3  |  17.6  | 32% | 50% |  58% |
|         VideoCoCa         |  ASR  |   0.8  |   20.3  |  9.2  |  21.9  | 36% | 53% |  59% |


### Summary Tasks
|           Model           | Audio | BLEU-4 | ROUGE-L | CIDEr | BLEURT | R@1 | R@5 | R@10 |
|:-------------------------:|:-----:|:------:|:-------:|:-----:|:------:|:---:|:---:|:----:|
|        Human (Avg)        |  Raw  |  15.7  |   34.5  |  36.9 |  55.6  |  -  |  -  |   -  |
|        Human (Min)        |  Raw  |  12.4  |   32.1  |  30.9 |  53.6  |  -  |  -  |   -  |
|       ImageBind-LLM       |  N/A  |   1.5  |   22.7  |  1.1  |  45.8  |  -  |  -  |   -  |
| Video-LLaMA2-Instruct 13B |  N/A  |   0.5  |   18.2  |  0.0  |  39.9  |  -  |  -  |   -  |
| Video-LLaMA2-Instruct 13B |  Raw  |   0.5  |   18.2  |  0.0  |  40.0  |  -  |  -  |   -  |
|  Video-LLaMA2-Instruct 7B |  N/A  |   0.5  |   19.1  |  0.0  |  43.9  |  -  |  -  |   -  |
|  Video-LLaMA2-Instruct 7B |  Raw  |   0.5  |   19.1  |  0.1  |  43.9  |  -  |  -  |   -  |
|        VideoChatGPT       |  N/A  |   2.9  |   24.4  |  5.8  |  46.7  |  -  |  -  |   -  |
|         VideoCoCa         |  N/A  |   0.9  |   16.4  |  3.3  |  23.9  | 25% | 41% |  48% |
|         VideoCoCa         |  ASR  |   2.0  |   21.6  |  5.5  |  22.9  | 27% | 42% |  48% |