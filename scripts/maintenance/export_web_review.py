"""Export a self-contained, read-only browser for web and textbook documents.

No model runs or extra dependencies: pandas/pyarrow are already used by the
analysis notebooks. Large generated artifacts stay on the persistent mount.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/mnt/tianzhu_hot/data-picking')
DATASET = 'math-web-retrieval-v2-4096-seed20261023'
ANALYSIS = '2026-10-23_095832Z_math-web-probe-qwen25-1p5b-n4096-b64-s64-seed42'
QUALITY_DATASET = 'math-document-quality-v1-seed20261024'
QUALITY_ANALYSIS = '2026-10-24_102053Z_math-document-quality-qwen25-1p5b-step64-seed20261024'
SCORES = ['ref_mean_nll', 'sft_mean_nll', 'delta_loss', 'relative_gain']
PROVENANCE = ['id', 'source', 'url', 'source_dataset', 'source_revision',
              'source_file', 'source_row', 'crawl', 'text_hash']
EXTRA_PROVENANCE = ['corpus', 'title', 'book', 'chapter', 'chapter_id', 'memberships',
                    'mathml_elements', 'images', 'quality_score', 'quality_int_score', 'source_subset']
CORPUS_NAMES = {'openstax': 'OpenStax', 'finemath': 'FineMath-4+',
                'openwebmath': 'OpenWebMath', 'background': 'FineWeb 背景（未标注）'}
QUALITY_SOURCES = {'openstax': 'openstax/osbooks-college-algebra-bundle',
                   'finemath': 'HuggingFaceTB/finemath'}


def verify_file(path, expected_hash, expected_size=None):
    content = path.read_bytes()
    if ((expected_size is not None and len(content) != expected_size)
            or hashlib.sha256(content).hexdigest() != expected_hash):
        raise ValueError(f'Input checksum mismatch: {path}')


def read_scores(path):
    if path.parent.name == 'snapshot':
        info = json.loads((path.parent.parent / 'manifest.json').read_text())['files'][path.name]
        verify_file(path, info['sha256'], info['bytes'])
    return pd.read_parquet(path)


def join_documents(documents, provenance, scores):
    """Reject incomplete/ambiguous joins rather than silently dropping pages."""
    for name, table in [('documents', documents), ('provenance', provenance), ('scores', scores)]:
        if table['id'].isna().any() or not table['id'].is_unique:
            raise ValueError(f'{name}: document IDs must be present and unique')
        if set(table['id']) != set(documents['id']):
            raise ValueError(f'{name}: document ID set differs from the body table')
    metadata_columns = PROVENANCE + [key for key in EXTRA_PROVENANCE if key in provenance]
    joined = documents[['id', 'text', 'token_count']].merge(
        provenance[metadata_columns], on='id', validate='one_to_one')
    joined = joined.merge(scores[['id', 'source', *SCORES]], on='id',
                          validate='one_to_one', suffixes=('', '_score'))
    if not joined['source'].eq(joined['source_score']).all():
        raise ValueError('Source labels disagree between provenance and scores')
    if not joined['source'].isin(['math', 'background']).all():
        raise ValueError('Unexpected web source')
    if joined[['text', *SCORES]].isna().any().any():
        raise ValueError('Missing body text or loss scores')
    if not joined[SCORES].map(math.isfinite).all().all():
        raise ValueError('Loss scores must be finite')
    if not joined['text'].map(lambda value: isinstance(value, str)).all():
        raise ValueError('Body text must be a string')
    return joined.drop(columns='source_score')


def join_quality_documents(documents, provenance, scores, original_scores):
    """Add only new positives; verify both repeated background copies first."""
    if set(documents['corpus']) != set(QUALITY_SOURCES):
        raise ValueError('Expected both OpenStax and FineMath documents')
    if (set(scores['comparison']) != set(QUALITY_SOURCES)
            or not scores['source'].isin(['math', 'background']).all()
            or scores.duplicated(['comparison', 'id']).any()):
        raise ValueError('Unexpected quality comparison or duplicated document IDs')
    background = original_scores.loc[original_scores['source'].eq('background')].set_index('id').sort_index()
    check_columns = ['source', *SCORES] + [key for key in ['token_count', 'scored_tokens'] if key in background]
    for corpus, comparison in scores.groupby('comparison'):
        positive = comparison.loc[comparison['source'].eq('math')]
        expected = documents.loc[documents['corpus'].eq(corpus)]
        if set(positive['id']) != set(expected['id']) or not positive['corpus'].eq(corpus).all():
            raise ValueError(f'{corpus}: scored document set or corpus label differs')
        repeated = comparison.loc[comparison['source'].eq('background')].set_index('id').sort_index()
        if not repeated['corpus'].eq('background').all():
            raise ValueError(f'{corpus}: unexpected background corpus label')
        try:
            pd.testing.assert_frame_equal(background[check_columns], repeated[check_columns],
                                          check_dtype=False, check_exact=True)
        except AssertionError as error:
            raise ValueError(f'{corpus}: background scores differ from the original pool') from error
    positive_scores = scores.loc[scores['source'].eq('math')]
    # Keep the original class label (math); corpus drives the four-way UI filter.
    metadata = provenance.copy()
    metadata['source_dataset'] = metadata['corpus'].map(QUALITY_SOURCES)
    metadata['source_subset'] = metadata['corpus'].map({'finemath': 'finemath-4plus', 'openstax': None})
    joined = join_documents(documents, metadata, positive_scores)
    if joined.set_index('id')['corpus'].to_dict() != documents.set_index('id')['corpus'].to_dict():
        raise ValueError('Corpus labels disagree between body and provenance')
    return joined


def embedded_json(value):
    """Keep untrusted webpage text from terminating the HTML data element."""
    return (json.dumps(value, ensure_ascii=False, allow_nan=False)
            .replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
            .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))


def export_review(web_dir, score_file, output_dir, seed=20261023, sample_count=10,
                  *, quality_dir=None, quality_score_file=None):
    if (quality_dir is None) != (quality_score_file is None):
        raise ValueError('Supply both quality_dir and quality_score_file')
    original_scores = read_scores(score_file)
    frame = join_documents(pd.read_parquet(web_dir / 'documents.parquet'),
                           pd.read_parquet(web_dir / 'document_manifest.parquet'),
                           original_scores)
    frame['corpus'] = frame['source'].map({'math': 'openwebmath', 'background': 'background'})
    if quality_dir is not None:
        complete = json.loads((quality_dir / 'COMPLETED.json').read_text())
        for name in ['documents.parquet', 'document_manifest.parquet']:
            verify_file(quality_dir / name, complete['files'][name])
        extra = join_quality_documents(pd.read_parquet(quality_dir / 'documents.parquet'),
                                       pd.read_parquet(quality_dir / 'document_manifest.parquet'),
                                       read_scores(quality_score_file), original_scores)
        frame = pd.concat([frame, extra], ignore_index=True)
    if not frame['id'].is_unique:
        raise ValueError('Document ID collision across review corpora')
    # pandas converts Arrow/numpy scalar types to portable JSON values.
    records = json.loads(frame.to_json(orient='records', force_ascii=False, double_precision=15))
    random.Random(seed).shuffle(records)
    payload = {'documents': records, 'seed': seed, 'scores_path': str(score_file),
               'web_dir': str(web_dir), 'source_counts': frame['source'].value_counts().to_dict(),
               'corpus_counts': frame['corpus'].value_counts().to_dict(), 'corpus_names': CORPUS_NAMES,
               'quality_dir': str(quality_dir) if quality_dir is not None else None,
               'quality_scores_path': str(quality_score_file) if quality_score_file is not None else None}
    template = Path(__file__).with_name('web_review.html').read_text()
    if template.count('__REVIEW_DATA__') != 1:
        raise ValueError('Expected exactly one template data marker')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'index.html').write_text(template.replace('__REVIEW_DATA__', embedded_json(payload)))
    for corpus in payload['corpus_counts']:
        samples = [row for row in records if row['corpus'] == corpus][:sample_count]
        sample_name = 'math' if corpus == 'openwebmath' else corpus
        with (output_dir / f'{sample_name}.samples.txt').open('w') as handle:
            handle.write(f'固定随机样本；corpus={corpus}；seed={seed}；{len(samples)} 篇；正文完整保留。\n'
                         '来源标签不是人工数学内容标签；样本不是按 loss 挑选。\n\n')
            for row in samples:
                header = {key: row.get(key) for key in ['id', 'source', 'corpus', 'title', 'book', 'chapter', 'url', 'token_count', *SCORES]}
                handle.write('=' * 80 + '\n' + json.dumps(header, ensure_ascii=False, indent=2)
                             + '\n\n' + row['text'] + '\n\n')
    summary = {key: value for key, value in payload.items() if key != 'documents'}
    summary.update(documents=len(records), samples_per_source=sample_count,
                   note='Read-only review export; no new labels or score changes.')
    (output_dir / 'review_manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--web-dir', type=Path, default=ROOT / 'data/processed' / DATASET / 'web')
    parser.add_argument('--scores', type=Path, default=REPO / 'experiments-analysis' / ANALYSIS / 'snapshot/per_document.parquet')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reviews' / DATASET)
    parser.add_argument('--quality-dir', type=Path, default=ROOT / 'data/processed' / QUALITY_DATASET)
    parser.add_argument('--quality-scores', type=Path, default=REPO / 'experiments-analysis' / QUALITY_ANALYSIS / 'snapshot/per_document.parquet')
    parser.add_argument('--original-only', action='store_true', help='Export only the original OpenWebMath / FineWeb pool')
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--sample-count', type=int, default=10)
    args = parser.parse_args()
    if args.sample_count < 0:
        parser.error('--sample-count must be nonnegative')
    result = export_review(args.web_dir, args.scores, args.output_dir, args.seed, args.sample_count,
                           quality_dir=None if args.original_only else args.quality_dir,
                           quality_score_file=None if args.original_only else args.quality_scores)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('REVIEW_HTML', args.output_dir / 'index.html')


if __name__ == '__main__':
    main()