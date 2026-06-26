#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
import sys
import os
import re
import glob
from collections import deque
import time
import logging
import math
from sklearn.metrics import f1_score
from deepgo.utils import FUNC_DICT, Ontology, NAMESPACES
from deepgo.metrics import compute_metrics

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Data folder')
@ck.option(
    '--ont', '-ont', default='mf', type=ck.Choice(['mf', 'bp', 'cc']),
    help='GO subontology')
@ck.option(
    '--model-name', '-m', required=True, help='Prediction model name')
@ck.option(
    '--test-data-name', '-td', default='test', type=ck.Choice(['test', 'nextprot']),
    help='Test data set name')
@ck.option(
    '--combine', '-c', default='avg', type=ck.Choice(['avg', 'min', 'max', 'wf1']),
    help='Combination strategy. "wf1" weights each model per GO term by its '
         'validation F1 score for that term.')
@ck.option(
    '--f1-threshold', '-ft', default=0.5,
    help='Score threshold used to binarize validation predictions when '
         'computing per-term F1 weights (only used with --combine wf1)')
@ck.option(
    '--n-models', '-nm', default=6,
    help='Top N models for semantic entailment')
def main(data_root, ont, model_name, test_data_name, combine, f1_threshold, n_models):
    train_data_file = f'{data_root}/{ont}/train_data.pkl'
    valid_data_file = f'{data_root}/{ont}/valid_data.pkl'
    test_data_file = f'{data_root}/{ont}/{test_data_name}_predictions_{model_name}_0.pkl'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    go = Ontology(f'{data_root}/go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    train_df = pd.concat([train_df, valid_df])
    test_df = pd.read_pickle(test_data_file)
    
    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = test_df['prop_annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go.calculate_ic(annotations + test_annotations)
    
    
    # Print IC values of terms
    ics = {}
    for term in terms:
        ics[term] = go.get_ic(term)

    n_terms = len(terms)
    top_models = get_top_models(ont, model_name, n_models)
    print(top_models)

    if combine == 'wf1':
        # Per-term, validation-F1-weighted average across models.
        valid_labels = build_label_matrix(valid_df, terms_dict, n_terms)
        weighted_sum = None
        simple_sum = None
        weight_total = None
        for i in top_models:
            test_df = pd.read_pickle(
                f'{data_root}/{ont}/{test_data_name}_predictions_{model_name}_{i}.pkl')
            preds_m = np.stack(test_df['preds'].values).reshape(-1, n_terms)
            term_f1 = get_term_f1(
                data_root, ont, model_name, i, valid_labels, f1_threshold)
            if weighted_sum is None:
                weighted_sum = np.zeros_like(preds_m, dtype=np.float64)
                simple_sum = np.zeros_like(preds_m, dtype=np.float64)
                weight_total = np.zeros(n_terms, dtype=np.float64)
            weighted_sum += preds_m * term_f1[None, :]
            simple_sum += preds_m
            weight_total += term_f1
        # Fall back to a uniform mean for terms where every model scores F1=0,
        # so those terms are not forced to zero by an all-zero weight.
        zero_mask = weight_total == 0
        eval_preds = np.empty_like(weighted_sum)
        nonzero = ~zero_mask
        eval_preds[:, nonzero] = weighted_sum[:, nonzero] / weight_total[nonzero][None, :]
        eval_preds[:, zero_mask] = simple_sum[:, zero_mask] / len(top_models)
        eval_preds = eval_preds.astype(np.float32)
    else:
        eval_preds = []
        for i in top_models: #range(6):#[0, 5, 6, 8]:
            #if i not in top_models:
            #    continue
            test_df = pd.read_pickle(f'{data_root}/{ont}/{test_data_name}_predictions_{model_name}_{i}.pkl')
            for j, row in enumerate(test_df.itertuples()):
                if j == len(eval_preds):
                    eval_preds.append(row.preds)
                else:
                    if combine == 'max':
                        eval_preds[j] = np.maximum(eval_preds[j], row.preds)
                    elif combine == 'min':
                        eval_preds[j] = np.minimum(eval_preds[j], row.preds)
                    elif combine == 'avg':
                        eval_preds[j] = eval_preds[j] + row.preds
                    else:
                        raise NotImplementedError()

        eval_preds = np.stack(eval_preds).reshape(-1, n_terms)
        if combine == 'avg':
            eval_preds /= len(top_models) # taking mean

    fmax, smin, tmax, wfmax, wtmax, avg_auc, aupr, avgic, fmax_spec_match = compute_metrics(
        test_df, go, terms_dict, terms, ont, eval_preds)

    print(ont)
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}, spec: {fmax_spec_match}')
    print(f'WFmax: {wfmax:0.3f}, threshold: {wtmax}')
    print(f'AUPR: {aupr:0.3f}')
    print(f'AVGIC: {avgic:0.3f}')


def build_label_matrix(df, terms_dict, n_terms):
    """Build a binary (n_proteins, n_terms) label matrix from prop_annotations."""
    labels = np.zeros((len(df), n_terms), dtype=np.int32)
    for i, row in enumerate(df.itertuples()):
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                labels[i, terms_dict[go_id]] = 1
    return labels


def get_term_f1(data_root, ont, model_name, i, valid_labels, threshold):
    """Per-term validation F1 for model `i`, returned as a (n_terms,) array.

    Reads the propagated validation predictions written by deepgo_test.py
    (`valid_<model>_<i>.valid_preds.pkl`) and scores them against valid_labels.
    """
    valid_preds = pd.read_pickle(
        f'{data_root}/{ont}/valid_{model_name}_{i}.valid_preds.pkl')
    valid_preds = np.stack(valid_preds).reshape(valid_labels.shape[0], -1)
    valid_preds_bin = (valid_preds >= threshold).astype(np.int32)
    return f1_score(valid_labels, valid_preds_bin, average=None, zero_division=0)


def get_top_models(ont, model, n_models):
    valid_losses = []
    # Discover the model indices that actually have a validation report on disk
    # instead of assuming a fixed range of 10 models.
    pf_files = sorted(glob.glob(f'data/{ont}/valid_{model}_*.pf'))
    pattern = re.compile(rf'valid_{re.escape(model)}_(\d+)\.pf$')
    for pf_file in pf_files:
        match = pattern.search(os.path.basename(pf_file))
        if match is None:
            continue
        ind = int(match.group(1))
        with open(pf_file) as f:
            lines = f.readlines()
            it = lines[-1].strip().split(', ')[0].split(' - ')
            loss = float(it[-1])
            valid_losses.append((ind, loss))
    valid_losses = sorted(valid_losses, key=lambda x: x[1])
    valid_losses = valid_losses[:n_models]
    result = [m_id for m_id, loss in valid_losses]
    print(valid_losses)
    return set(result)

if __name__ == '__main__':
    main()
