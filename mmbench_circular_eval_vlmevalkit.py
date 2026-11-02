# ReGainDrop release preparation (2026-11-02): distribution notice added.
# Implementation retained from the author-supplied development snapshot; see NOTICE.
#!/usr/bin/env python3
"""MMBench circular eval (VLMEvalKit-compatible exact matching)."""
from __future__ import annotations

import argparse
import copy as cp
import json
import os
import string

import pandas as pd


def build_choices(item) -> dict:
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in item and (not pd.isna(item[ch])):
            ret[ch] = item[ch]
    return ret


def can_infer_option(answer, choices: dict):
    if 'Failed to obtain answer via API' in answer:
        return False
    reject_to_answer = [
        "Sorry, I can't help with images of people yet.", "I can't process this file.",
        "I'm sorry, but without the image provided", 'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, chs, prefix='', suffix=''):
        cnt = 0
        for c in chs:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    for c in '.()[],:;!*#{}':
        answer_mod = answer_mod.replace(c, ' ')
    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)
    if count == 1:
        for ch in choices:
            if ch in splits and splits.index(ch) > (len(splits) - 5):
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False


def can_infer_text(answer, choices: dict):
    answer = answer.lower()
    if len(answer) > 2 * sum(len(str(v)) for v in choices.values()):
        return False
    chs = {k: str(choices[k]).lower() for k in choices}
    cands = []
    for k in chs:
        if chs[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices: dict):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


def prefetch_answer(item) -> str | bool:
    choices = build_choices(item)
    return can_infer(item['prediction'], choices)


def prefetch_circular_group(sub_data: pd.DataFrame, verbose: bool = False):
    lt = len(sub_data)
    GT, PRED = [], []
    for i in range(lt):
        item = sub_data.iloc[i]
        GT.append(item['GT'])
        PRED.append(prefetch_answer(item))
        if PRED[-1] and (GT[-1] != PRED[-1]):
            log = (
                f'Failed in Prefetching Rolling {i}: Answer is {GT[-1]}, '
                f"Prediction is {item['prediction']}, Pre-fetched is {PRED[-1]}. "
            )
            return dict(hit=0, log=log)
    flag = True
    for g, p in zip(GT, PRED):
        if g != p:
            flag = False
    ret = (dict(hit=1, log='Succeed During Pre-fetching'),) if flag else (None,)
    ret = ret + (GT, PRED) if verbose else ret
    return ret if len(ret) > 1 else ret[0]


def extract_answer_no_llm(item) -> dict:
    choices = build_choices(item)
    ret = can_infer(item['prediction'], choices)
    if ret:
        return dict(opt=ret, log=item['prediction'])
    return dict(opt='Z', log='Failed in Prefetch, exact matching only.')


def eval_circular_group(_model, sub_data: pd.DataFrame) -> dict:
    prefetched = prefetch_circular_group(sub_data, verbose=True)
    if isinstance(prefetched, dict) and 'hit' in prefetched:
        return prefetched
    res, GT, PRED = prefetched
    if res is not None:
        return res
    lt = len(sub_data)
    log = ''
    PRED = list(PRED)
    for i in range(lt):
        if PRED[i]:
            log += f'Rolling {i} Matched.\n'
        else:
            ex = extract_answer_no_llm(sub_data.iloc[i])
            opt, match_log = ex['opt'], ex['log']
            PRED[i] = opt
            if PRED[i] != GT[i]:
                log += (
                    f"Failed in Rolling {i}: Answer is {GT[i]}; Prediction is {sub_data.iloc[i]['prediction']}; "
                    f'Pre-fetched is {PRED[i]}; Match Log is {match_log}.\n'
                )
                return dict(hit=0, log=log)
            log += (
                f"Rolling {i}: Answer is {GT[i]}, Prediction is {sub_data.iloc[i]['prediction']}, "
                f'Pre-fetched is {PRED[i]}.\n'
            )
    return dict(hit=1, log=log)


def mcq_circular_eval_local(data: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}
    data = data[data['index'].isin(answer_map)].copy()
    data['GT'] = [answer_map[idx] for idx in data['index']]
    if 'g_index' not in data:
        data['g_index'] = [int(x % 1_000_000) for x in data['index']]
    data_main = data[data['index'] == data['g_index']].copy()
    results = {}
    for idx in data_main['index']:
        sub = data[data['g_index'] == idx]
        results[idx] = eval_circular_group(None, sub)
    out = data_main.copy()
    out['hit'] = [results[i]['hit'] for i in out['index']]
    out['log'] = [results[i]['log'] for i in out['index']]
    out = out.drop(columns=['GT'], errors='ignore')
    return out


def report_acc(df: pd.DataFrame) -> pd.DataFrame:
    from collections import defaultdict
    import numpy as np

    MMB_abbrs = {
        'coarse_perception': 'CP',
        'finegrained_perception (instance-level)': 'FP-S',
        'finegrained_perception (cross-instance)': 'FP-C',
        'logic_reasoning': 'LR',
        'relation_reasoning': 'RR',
        'attribute_reasoning': 'AR'
    }
    res = defaultdict(list)
    if 'split' in df:
        splits = list(set(df['split']))
        res['split'] = splits
    else:
        df = df.copy()
        df['split'] = ['none'] * len(df)
        res['split'] = ['none']
    for group in [None, 'l2-category', 'category']:
        if group is None:
            res['Overall'] = [np.mean(df[df['split'] == sp]['hit']) for sp in res['split']]
        elif group not in df:
            continue
        else:
            abilities = sorted(set(df[group]))
            for ab in abilities:
                ab_name = MMB_abbrs[ab] if ab in MMB_abbrs else ab
                sub_df = df[df[group] == ab]
                res[ab_name] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]
    return pd.DataFrame(res)


def load_jsonl_preds(path: str) -> dict:
    preds = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            preds[r['question_id']] = str(r['text']).strip()
    return preds


def main():
    p = argparse.ArgumentParser(description='MMBench circular eval (VLMEvalKit-compatible exact matching).')
    p.add_argument('--annotation', required=True, help='Official MMBench .tsv (dev, with answer column).')
    p.add_argument('--jsonl', required=True, help='LLaVA model_vqa_mmbench output .jsonl')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--tag', default='mmbench_eval', help='Output file prefix')
    args = p.parse_args()

    meta = pd.read_table(args.annotation)
    preds = load_jsonl_preds(args.jsonl)
    eval_df = meta.copy()
    eval_df['prediction'] = eval_df['index'].map(preds)
    missing = int(eval_df['prediction'].isna().sum())
    if missing:
        raise SystemExit(f'Missing predictions for {missing} rows (jsonl index vs tsv index).')

    os.makedirs(args.out_dir, exist_ok=True)
    pred_fmt = os.path.join(args.out_dir, f'{args.tag}_with_prediction.xlsx')
    eval_df.to_excel(pred_fmt, index=False, engine='openpyxl')

    scored = mcq_circular_eval_local(eval_df, meta)
    acc = report_acc(scored)
    acc_path = os.path.join(args.out_dir, f'{args.tag}_acc.csv')
    acc.to_csv(acc_path, index=False)
    print(f'Wrote VLMEvalKit-style prediction table: {pred_fmt}')
    print(f'Wrote accuracy report: {acc_path}')
    overall = acc['Overall'].iloc[0] if 'Overall' in acc.columns else float('nan')
    print(f'Overall hit rate (circular, exact matching): {overall:.4f}')


if __name__ == '__main__':
    main()
