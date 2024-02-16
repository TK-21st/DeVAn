"""Compute BLEURT Score

Please follow instruction in https://github.com/lucadiliello/bleurt-pytorch to setup bleurt_pytorch first.
"""
from tqdm.auto import tqdm
import pandas as pd
import json
import os
from random import sample
import numpy as np

import pandas as pd
import gc
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

if __name__ == "__main__":
    # Load Model
    config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20')
    model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20')
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')

    model = model.cuda()
    model.eval()

    # Load Data
    df = pd.read_parquet("data_with_predictions.parquet")

    # predictions of all models are assumed to have name 'caption-model_name' or 'summary-model_name'
    pred_columns = [c for c in df.columns if c.startswith('caption-') or c.startswith('summary-')]

    bs = 1000 # batchsize of bleurt computation
    score_total = 0
    for col in pred_columns:
        print(f"Running ... {col}")
        pred = df[col].dropna()

        # references are assumed to be named caption1...caption5 and summary1...summary5
        if col.startswith('caption-'):
            gt = df[[f'caption{n}' for n in range(1,6)]].dropna(how='all', axis=0)
        else:
            gt = df[[f'summary{n}' for n in range(1,6)]].dropna(how='all', axis=0)
        common_keys = sorted(list(set(pred.index).intersection(set(gt.index))))
        pred = pred.loc[common_keys].to_frame(name='pred').reset_index()
        gt = gt.loc[common_keys]
        references = gt.reset_index().melt(id_vars='index', value_name='gt').drop(columns='variable')
        _df = pd.merge(references, pred, on='index')
        df_res = []
        
        for _, _ddf in tqdm(_df.groupby(np.arange(len(_df)) // bs), total=len(_df)//bs): 
            with torch.no_grad():
                inputs = tokenizer(_ddf['gt'].tolist(), _ddf['pred'].tolist(), padding='longest', return_tensors='pt', truncation=True, max_length=512)
                i_id = inputs['input_ids'].cuda()
                i_mask =inputs['attention_mask'].cuda()
                result = model(input_ids=i_id, attention_mask=i_mask)
                scores = result.logits.flatten().tolist()
                _ddf.loc[:, "bleurt"] = scores
                df_res.append(_ddf[['index', 'bleurt']].rename(columns={"bleurt": f"bleurt-{col}"}))
            torch.cuda.empty_cache()
            gc.collect()
        df_res = pd.concat(df_res)
        df_res.to_json(f"bleurt_{col}.json")
