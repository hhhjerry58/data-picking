import pytest
from data_picking.document_sanity import compare_document_scores


def row(name,count,total):
    return {'id':name,'document_id':'doc','scored_tokens':count,'sum_nll':total,
            'mean_nll':total/count,'tokenization_hash':name}


def test_tiny_tail_is_weighted_by_actual_document_tokens():
    single=[row('long',2047,4094.),row('tail',1,2.)]
    batch=[row('long',2047,4094.),row('tail',1,2.+2.1e-5)]
    result=compare_document_scores(batch,batch,single,2e-5)
    assert result['window_batch_padding_max_nll_difference']>2e-5
    assert result['batch_padding_max_nll_difference']<1.1e-8


def test_a_truly_short_document_still_must_meet_the_same_tolerance():
    single=[row('only',1,2.)]
    batch=[row('only',1,2.+3e-5)]
    with pytest.raises(RuntimeError,match='Document NLL sanity failure'):
        compare_document_scores(batch,batch,single,2e-5)
