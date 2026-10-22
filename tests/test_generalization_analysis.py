import pandas as pd
import numpy as np

import data_picking.generalization_analysis as analysis

from data_picking.generalization_analysis import LOSS_SCORES, GRADIENT_SCORES, evaluate_pair, match_three_domains


def examples():
    rows=[]
    for domain, offset in [('A',0),('B',10),('C',20)]:
        for i in range(8):
            rows.append({'id':f'{domain}{i}','domain':domain,'group_id':f'{domain}{i//2}',
                         'prompt_tokens':40+i,'response_tokens':80+i,
                         **{key:float(offset+i) for key in LOSS_SCORES+GRADIENT_SCORES}})
    return pd.DataFrame(rows)


def test_three_way_matching_preserves_equal_counts_and_unique_ids():
    frame=examples()
    matched,summary=match_three_domains(frame,42)
    assert matched['id'].is_unique
    assert matched.groupby('domain').size().tolist()==[8,8,8]
    assert summary['retained_fraction']==1
    assert len(frame)==24


def test_b_vs_c_positive_class_and_paired_identical_baselines():
    result=evaluate_pair(examples(),'B','C',{'analysis':{'bootstrap_samples':20,'bootstrap_seed':42}})
    assert result['counts']=={'B':8,'C':8}
    assert result['loss_scores']['delta_loss']['roc_auc']==0
    assert result['gradient_scores']['ref_grad_l2']['roc_auc']==0
    assert result['paired_delta_auc_minus_baseline']['neg_ref_loss']=={'difference':0.,'ci95':[0.,0.]}


def test_bootstrap_resamples_entire_question_groups(monkeypatch):
    frame = examples()
    for key in LOSS_SCORES + GRADIENT_SCORES:
        frame[key] = 1000 + np.arange(len(frame))
    real_auc = analysis.auc
    observed = []

    def checked_auc(labels, scores):
        values = np.asarray(scores)
        if np.all(values >= 1000):
            counts = np.bincount((values - 1000).astype(int), minlength=len(frame))
            # Every pair shares a question. Even in bootstrap replicates, its
            # answers must occur equally often; row-wise sampling violates this.
            assert np.array_equal(counts[::2], counts[1::2])
            observed.append(counts)
        return real_auc(labels, scores)

    monkeypatch.setattr(analysis, 'auc', checked_auc)
    result = evaluate_pair(frame, 'B', 'C', {'analysis': {'bootstrap_samples': 20, 'bootstrap_seed': 42}})
    assert len(observed) == 14 * 21
    assert any((counts > 1).any() for counts in observed)
    assert result['question_group_counts'] == {'B': 4, 'C': 4}
