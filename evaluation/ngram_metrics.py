"""Compute N-Gram Based metrics

Place this file in the root path of https://github.com/jmhessel/clipscore.
"""
from tqdm.auto import tqdm
import pandas as pd
import json
import generation_eval_utils

def compute_scores(image_ids, candidates, references):
    candidates = [candidates[cid] for cid in image_ids]
    references = [references[cid] for cid in image_ids]
    if isinstance(references[0], str):
        references = [[r] for r in references]

    
    other_metrics = generation_eval_utils.get_all_metrics(references, candidates, return_per_cap=True)
    metrics = dict()
    for k, v in other_metrics.items():
        if k == 'bleu':
            for bidx, sc in enumerate(v):
                metrics[f'BLEU-{bidx+1}'] = {cid:_v for cid, _v in zip(image_ids, sc)}
        else:
            metrics[k.upper()] = {cid:_v for cid, _v in zip(image_ids, v)}
    return metrics

if __name__ == "__main__":
    df = pd.read_parquet("data_with_predictions.parquet")

    # predictions of all models are assumed to have name 'caption-model_name' or 'summary-model_name'
    pred_columns = [c for c in df.columns if c.startswith('caption-') or c.startswith('summary-')]

    results = []
    for col in tqdm(pred_columns): # for each model

        task = col.split('-')[0] # caption/summary
        model = '-'.join(col.split('-')[1:]) # model name

        # references are assumed to be named caption1...caption5 and summary1...summary5
        if task == 'caption': 
            ref_cols = [f'caption{n}' for n in range(1,6)]
        elif task == 'summary':
            ref_cols = [f'summary{n}' for n in range(1,6)]
        df_col = df[[col]+ref_cols]
        df_col = df_col.dropna(how='any', axis=0)

        # find candidates and references with matching video_ids
        candidates = {key: list(val.values())[0] for key,val in df_col[[col]].to_dict(orient='index').items()}
        references = {key: list(val.values()) for key,val in  df_col[ref_cols].to_dict(orient='index').items()}
        common_keys = sorted(list(set(candidates.keys()).intersection(set(references.keys()))))
        
        metric = compute_scores(common_keys, candidates, references)
        metric['model'] = model
        metric['task'] = task
        results.append(metric)

    pd.DataFrame(results).to_json("output_file_name.json")