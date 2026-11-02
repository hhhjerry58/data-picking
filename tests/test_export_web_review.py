"""The review export preserves bodies and never executes their markup."""
import hashlib
import json
from pathlib import Path
import runpy

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/maintenance/export_web_review.py'
REVIEW = runpy.run_path(str(SCRIPT))


def tables():
    documents = pd.DataFrame([
        {'id': 'm', 'text': '</script><script>alert(1)</script> 数学\n$x^2$ & <b>x</b>', 'token_count': 20},
        {'id': 'b', 'text': 'ordinary page\nsecond line', 'token_count': 6},
    ])
    provenance = pd.DataFrame([
        {key: {'id': identifier, 'source': source, 'source_row': index}.get(key, 'example')
         for key in REVIEW['PROVENANCE']}
        for index, (identifier, source) in enumerate([('m', 'math'), ('b', 'background')])
    ])
    scores = pd.DataFrame([
        {'id': identifier, 'source': source, 'ref_mean_nll': 2., 'sft_mean_nll': 1.,
         'delta_loss': 1., 'relative_gain': .5}
        for identifier, source in [('m', 'math'), ('b', 'background')]
    ])
    return documents, provenance, scores


def test_join_uses_ids_not_row_order():
    documents, provenance, scores = tables()
    result = REVIEW['join_documents'](documents, provenance.iloc[::-1], scores.iloc[::-1])
    assert result.set_index('id').loc['m', 'source'] == 'math'
    assert result.set_index('id').loc['b', 'text'] == 'ordinary page\nsecond line'


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'source', 'nonfinite'])
def test_rejects_inconsistent_inputs(problem):
    documents, provenance, scores = tables()
    if problem == 'missing':
        scores = scores.iloc[:1]
    elif problem == 'duplicate':
        scores.loc[1, 'id'] = 'm'
    elif problem == 'source':
        scores.loc[0, 'source'] = 'background'
    else:
        scores.loc[0, 'delta_loss'] = float('inf')
    with pytest.raises(ValueError):
        REVIEW['join_documents'](documents, provenance, scores)


def test_embedded_json_does_not_terminate_script():
    original = {'text': '</ScRiPt><script>alert(1)</script> & 数学\u2028\u2029'}
    encoded = REVIEW['embedded_json'](original)
    assert '<' not in encoded and '&' not in encoded
    assert json.loads(encoded) == original


def test_export_preserves_full_bodies_and_is_repeatable(tmp_path):
    documents, provenance, scores = tables()
    web = tmp_path / 'web'
    web.mkdir()
    documents.to_parquet(web / 'documents.parquet')
    provenance.to_parquet(web / 'document_manifest.parquet')
    score_file = tmp_path / 'scores.parquet'
    scores.to_parquet(score_file)
    output = tmp_path / 'review'
    result = REVIEW['export_review'](web, score_file, output, sample_count=1)
    first = (output / 'index.html').read_text()
    payload_text = first.split('<script id="payload" type="application/json">', 1)[1].split('</script>', 1)[0]
    payload = json.loads(payload_text)
    exported = {row['id']: row['text'] for row in payload['documents']}
    assert exported == documents.set_index('id')['text'].to_dict()
    assert result['source_counts'] == {'math': 1, 'background': 1}
    assert documents.iloc[0]['text'] in (output / 'math.samples.txt').read_text()
    REVIEW['export_review'](web, score_file, output, sample_count=1)
    assert (output / 'index.html').read_text() == first


def quality_tables():
    _, _, original_scores = tables()
    documents = pd.DataFrame([
        {'id': 'o', 'corpus': 'openstax', 'text': 'Functions\n$x^2$ 教材全文', 'token_count': 9},
        {'id': 'f', 'corpus': 'finemath', 'text': 'FineMath\n</script> 数学全文', 'token_count': 8},
    ])
    provenance = pd.DataFrame([
        {key: {'id': identifier, 'source': 'math', 'source_row': 0}.get(key, 'example')
         for key in REVIEW['PROVENANCE'] if key != 'source_dataset'}
        for identifier in ['o', 'f']
    ])
    provenance['corpus'] = ['openstax', 'finemath']
    provenance['title'] = ['Functions', None]
    provenance['book'] = ['Precalculus 2e', None]
    provenance['chapter'] = ['Functions chapter', None]
    provenance['quality_score'] = [None, 4.5]
    parts = []
    for identifier, corpus in [('o', 'openstax'), ('f', 'finemath')]:
        part = original_scores.copy()
        part.loc[part['id'].eq('m'), 'id'] = identifier
        part['corpus'] = [corpus, 'background']
        part['comparison'] = corpus
        parts.append(part)
    return documents, provenance, pd.concat(parts, ignore_index=True), original_scores


def test_quality_join_preserves_titles_and_separates_corpora():
    documents, provenance, scores, original = quality_tables()
    result = REVIEW['join_quality_documents'](documents, provenance.iloc[::-1], scores.iloc[::-1], original)
    assert result['id'].tolist() == ['o', 'f']
    assert result['source'].tolist() == ['math', 'math']
    assert result['corpus'].tolist() == ['openstax', 'finemath']
    assert result.loc[0, 'title'] == 'Functions'
    assert result.loc[0, 'book'] == 'Precalculus 2e'
    assert result.loc[1, 'quality_score'] == 4.5
    assert result.loc[1, 'source_dataset'] == 'HuggingFaceTB/finemath'
    assert result.loc[1, 'source_subset'] == 'finemath-4plus'


@pytest.mark.parametrize('problem', [
    'background_score', 'missing_background', 'duplicate_background',
    'missing_positive', 'wrong_comparison', 'wrong_corpus', 'wrong_source', 'wrong_provenance',
])
def test_quality_join_rejects_misaligned_comparisons(problem):
    documents, provenance, scores, original = quality_tables()
    if problem == 'background_score':
        scores.loc[1, 'ref_mean_nll'] += .1
    elif problem == 'missing_background':
        scores = scores.drop(index=1)
    elif problem == 'duplicate_background':
        scores = pd.concat([scores, scores.iloc[[1]]], ignore_index=True)
    elif problem == 'missing_positive':
        scores = scores.drop(index=0)
    elif problem == 'wrong_comparison':
        scores.loc[0, 'comparison'] = 'other'
    elif problem == 'wrong_corpus':
        scores.loc[0, 'corpus'] = 'finemath'
    elif problem == 'wrong_source':
        scores.loc[0, 'source'] = 'background'
    else:
        provenance.loc[0, 'corpus'] = 'finemath'
    with pytest.raises(ValueError):
        REVIEW['join_quality_documents'](documents, provenance, scores, original)


def write_review_inputs(tmp_path):
    docs, provenance, scores = tables()
    web = tmp_path / 'web'
    web.mkdir()
    docs.to_parquet(web / 'documents.parquet')
    provenance.to_parquet(web / 'document_manifest.parquet')
    scores.to_parquet(tmp_path / 'scores.parquet')
    new_docs, new_provenance, new_scores, _ = quality_tables()
    quality = tmp_path / 'quality'
    quality.mkdir()
    new_docs.to_parquet(quality / 'documents.parquet')
    new_provenance.to_parquet(quality / 'document_manifest.parquet')
    new_scores.to_parquet(tmp_path / 'quality_scores.parquet')
    files = {name: hashlib.sha256((quality / name).read_bytes()).hexdigest()
             for name in ['documents.parquet', 'document_manifest.parquet']}
    (quality / 'COMPLETED.json').write_text(json.dumps({'files': files}))
    return web, quality


def test_four_corpus_export_has_one_background_and_complete_bodies(tmp_path):
    web, quality = write_review_inputs(tmp_path)
    output = tmp_path / 'review'
    result = REVIEW['export_review'](web, tmp_path / 'scores.parquet', output, sample_count=1,
                                    quality_dir=quality, quality_score_file=tmp_path / 'quality_scores.parquet')
    page = (output / 'index.html').read_text()
    payload = json.loads(page.split('<script id="payload" type="application/json">', 1)[1].split('</script>', 1)[0])
    records = {row['id']: row for row in payload['documents']}
    assert len(payload['documents']) == len(records) == 4
    assert result['corpus_counts'] == {'openwebmath': 1, 'background': 1, 'openstax': 1, 'finemath': 1}
    assert result['source_counts'] == {'math': 3, 'background': 1}
    assert records['o']['title'] == 'Functions'
    original_docs, _, original_scores = tables()
    extra_docs, _, _, _ = quality_tables()
    for _, row in pd.concat([original_docs, extra_docs]).iterrows():
        assert records[row['id']]['text'] == row['text']
    for _, row in original_scores.iterrows():
        for key in REVIEW['SCORES']:
            assert records[row['id']][key] == row[key]
    for filename, identifier in [('math', 'm'), ('background', 'b'), ('openstax', 'o'), ('finemath', 'f')]:
        assert records[identifier]['text'] in (output / f'{filename}.samples.txt').read_text()


def test_changed_quality_body_fails_completion_hash(tmp_path):
    web, quality = write_review_inputs(tmp_path)
    (quality / 'documents.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        REVIEW['export_review'](web, tmp_path / 'scores.parquet', tmp_path / 'output',
                                quality_dir=quality, quality_score_file=tmp_path / 'quality_scores.parquet')


def test_snapshot_scores_checked_before_reading(tmp_path):
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    path = snapshot / 'per_document.parquet'
    tables()[2].to_parquet(path)
    info = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / 'manifest.json').write_text(json.dumps({'files': {path.name: info}}))
    assert len(REVIEW['read_scores'](path)) == 2
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        REVIEW['read_scores'](path)