"""CPU-only checkpoint distributions, emitted as JSON for a notebook wrapper.

Run with the existing project venv; never install packages or load a model.
Reads only checksummed repository snapshots and writes no experiment artifacts.
"""
import base64
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/dev/shm/data-picking-runtime/matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAIN = '2026-10-23_095832Z_math-web-probe-qwen25-1p5b-n4096-b64-s64-seed42'
DIAGNOSTIC = '2026-10-23_165639Z_math-web-probe-diagnostics-qwen25-1p5b-seed42'
STEPS = [16, 32, 48, 64]
KEYS = ['delta_loss', 'relative_gain']
COLORS = {'math': '#1674B8', 'background': '#DF7931'}
NAMES = {'math': 'Injected OpenWebMath', 'background': 'Unlabeled CC background'}
RELATIVE_RANGE = (-.25, .15)


def checked_table(folder, name):
    manifest = json.loads((folder / 'manifest.json').read_text())
    path = folder / 'snapshot' / name
    info = manifest['files'][name]
    content = path.read_bytes()
    if len(content) != info['bytes'] or hashlib.sha256(content).hexdigest() != info['sha256']:
        raise ValueError(f'Snapshot checksum failed: {path}')
    return pd.read_parquet(path)


def load_scores(repo):
    base = repo / 'experiments-analysis'
    web = checked_table(base / DIAGNOSTIC, 'web_per_document.parquet')
    formal = checked_table(base / MAIN, 'per_document.parquet').set_index('id')
    assert sorted(web['checkpoint_step'].unique()) == STEPS
    assert not web.duplicated(['checkpoint_step', 'id']).any()
    assert np.isfinite(web[['ref_mean_nll', 'sft_mean_nll', *KEYS]].to_numpy()).all()
    for step, group in web.groupby('checkpoint_step'):
        assert group.groupby('source').size().to_dict() == {'background': 9000, 'math': 1000}
        assert set(group['id']) == set(formal.index)
        aligned = group.set_index('id').loc[formal.index]
        for key in ['source', 'token_count', 'scored_tokens']:
            assert aligned[key].equals(formal[key]), (step, key)
        np.testing.assert_allclose(aligned['ref_mean_nll'], formal['ref_mean_nll'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(group['delta_loss'], group['ref_mean_nll'] - group['sft_mean_nll'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(group['relative_gain'], group['delta_loss'] / np.maximum(group['ref_mean_nll'], 1e-6), rtol=0, atol=1e-12)
        if step == 64:
            for key in ['sft_mean_nll', *KEYS]:
                np.testing.assert_allclose(aligned[key], formal[key], rtol=0, atol=1e-12)
    return web


def build_payload(repo):
    web = load_scores(repo)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    lo = np.floor(web['delta_loss'].min() * 100) / 100
    hi = np.ceil(web['delta_loss'].max() * 100) / 100
    bins = {'delta_loss': np.linspace(lo, hi, 51),
            'relative_gain': np.linspace(*RELATIVE_RANGE, 51)}
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex='col', sharey='col')
    records = []
    for row, step in enumerate(STEPS):
        group = web.loc[web['checkpoint_step'].eq(step)]
        for source, values in group.groupby('source'):
            records.append({'step': step, 'source': source, 'documents': len(values),
                            'mean_delta_loss': values['delta_loss'].mean(),
                            'mean_relative_gain': values['relative_gain'].mean(),
                            'fraction_improved': values['delta_loss'].gt(0).mean(),
                            'relative_outside_view': int((~values['relative_gain'].between(*RELATIVE_RANGE)).sum())})
        for col, key in enumerate(KEYS):
            ax = axes[row, col]
            for source, values in group.groupby('source'):
                # Normalize by ALL documents of this source, not only visible ones.
                weights = np.full(len(values), 1 / len(values))
                counts, edges = np.histogram(values[key], bins=bins[key], weights=weights)
                density = counts / np.diff(edges)
                assert np.isclose(np.sum(density * np.diff(edges)), values[key].between(edges[0], edges[-1]).mean())
                ax.bar(edges[:-1], density, width=np.diff(edges), align='edge', alpha=.5,
                       color=COLORS[source], label=NAMES[source], linewidth=0)
            ax.axvline(0, color='black', linestyle='--', linewidth=.9, label='Step 0: all changes = 0')
            ax.set(xlim=(bins[key][0], bins[key][-1]), xlabel=key, ylabel='Density',
                   title=f'Step {step} vs step 0 | ' + ('Reference NLL - SFT NLL' if col == 0 else 'Relative loss reduction'))
            ax.tick_params(axis='x', labelbottom=True)
            ax.legend(fontsize=7, loc='upper left')
    for col in range(2):
        ymax = max(ax.get_ylim()[1] for ax in axes[:, col])
        for ax in axes[:, col]:
            ax.set_ylim(0, ymax)
    fig.suptitle('Same 10,000 documents | Four checkpoints vs reference (step 0)', fontsize=14, y=.995)
    fig.tight_layout(rect=(0, .025, 1, .98))
    fig.text(.5, .008, 'Shared bins and axes within each metric. Relative view: [-0.25, 0.15]; tails remain in all statistics.',
             ha='center', fontsize=9)
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=130)
    plt.close(fig)
    # Step 0 is a mathematical identity, not a new measurement.
    baseline = [{'step': 0, 'source': source, 'documents': count,
                 'mean_delta_loss': 0., 'mean_relative_gain': 0.,
                 'fraction_improved': 0., 'relative_outside_view': 0}
                for source, count in [('background', 9000), ('math', 1000)]]
    summary = pd.DataFrame(baseline + records)
    return {'image_png_base64': base64.b64encode(buffer.getvalue()).decode('ascii'),
            'summary': summary.to_dict(orient='records'),
            'summary_html': summary.round(6).to_html(index=False),
            'validated_steps': STEPS, 'panels': len(axes.ravel()),
            'relative_view': list(RELATIVE_RANGE),
            'source_snapshot': str(Path('experiments-analysis') / DIAGNOSTIC / 'snapshot/web_per_document.parquet')}


if __name__ == '__main__':
    print(json.dumps(build_payload(Path(__file__).resolve().parents[2]), allow_nan=False))