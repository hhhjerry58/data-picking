import numpy as np
import pytest
import torch

from data_picking.document_utils import document_windows
from data_picking.utility_data import token_batches
from data_picking.utility_training import verify_cursor
from data_picking.loss import per_example_nll
from data_picking.utility_analysis import summarize_arrays


def test_token_budget_preserves_context_and_never_repeats_targets():
    windows=list(document_windows('one',list(range(25)),9))+list(document_windows('two',list(range(30,55)),9))
    rows,batches=token_batches(windows,42,5,7)
    source={w['id']:w for w in windows};seen=set()
    for batch in batches:assert sum(rows[i]['response_tokens'] for i in batch)==7
    assert sum(r['response_tokens'] for r in rows)==35
    for row in rows:
        w=source[row['source_window_id']]
        assert row['input_ids']==w['input_ids'][:len(row['input_ids'])]
        assert sum(row['response_mask'])==row['source_token_end']-row['source_token_start']
        for token in range(row['source_token_start'],row['source_token_end']):
            assert (row['document_id'],token) not in seen;seen.add((row['document_id'],token))
    assert any(r['response_mask'].index(1)>1 for r in rows)
    assert (rows,batches)==token_batches(windows,42,5,7)
    with pytest.raises(ValueError):token_batches(windows,42,100,7)


def test_token_weighted_gradient_is_invariant_to_microbatch_boundaries():
    torch.manual_seed(42)
    logits=torch.randn(2,5,7,requires_grad=True)
    tokens=torch.tensor([[0,1,2,3,4],[0,2,4,0,0]])
    mask=torch.tensor([[0,1,1,1,1],[0,1,0,0,0]],dtype=torch.bool)
    sums,means,counts=per_example_nll(logits,tokens,mask)
    (sums.sum()/counts.sum()).backward();expected=logits.grad.clone()
    logits.grad=None
    for i in range(2):
        s,_,_=per_example_nll(logits[i:i+1],tokens[i:i+1],mask[i:i+1]);(s.sum()/5).backward()
    torch.testing.assert_close(logits.grad,expected)


def test_resume_cursor_rejects_changed_batches_or_token_count():
    batches=[[0,1],[2,3]];s={'global_step':1,'next_data_step':1,'orders':[[0,1,2,3]],'batches':batches,'seen_response_tokens':7}
    verify_cursor(s,batches,4,7)
    for bad in [{**s,'seen_response_tokens':8},{**s,'next_data_step':0},{**s,'batches':[[0,2],[1,3]]}]:
        with pytest.raises(RuntimeError):verify_cursor(bad,batches,4,7)


def test_direct_utility_contrast_sign_and_paired_bootstrap():
    ref=np.array([1.,2.,3.,4.]);losses=np.array([[ref-.1,ref-.3]]*3)
    result=summarize_arrays(ref,losses,['a','b','c','d'],[42,43,44],['general','math'],50,42)
    assert result['math']['scores']['delta_loss']['mean']==pytest.approx(.3)
    assert result['math']['contrast_vs_general']['delta_loss']['mean']==pytest.approx(.2)
    assert result['math']['contrast_vs_general']['delta_loss']['hierarchical_ci95']==pytest.approx([.2,.2])
    assert result['general']['contrast_vs_general']['relative_gain']['hierarchical_ci95']==[0.,0.]
