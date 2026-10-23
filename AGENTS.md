# Project context

- This repository studies data selection for reducing loss on a target domain.
- The initial pilot splits domain A into A1 and A2, fine-tunes a reference model
  on A1, and measures whether changes in per-example loss distinguish unseen A2
  examples from domain B examples.
- Domain discrimination is a proxy experiment. It does not by itself establish
  which examples provide the largest improvement when used for further training.
- The user has approved the experiment design in `docs/pilot_design.md` and
  authorized training on local GPUs. The first pair is MATH Algebra (A) and
  Number Theory (B), with A1=1,024 and A2=B=512; see `configs/pilot.yaml`.
- The implementation is in `src/data_picking/` and `scripts/`. Source
  `scripts/env.sh` and use `.venv/bin/python`. Select a free GPU explicitly with
  `CUDA_VISIBLE_DEVICES`; do not disturb other users' GPU processes.
- Use Chinese for project discussion and review documents unless requested otherwise.

# Experiment conventions

- Keep A2 and B out of training and checkpoint selection. Freeze metric
  definitions, score directions, and the checkpoint rule before examining results.
- Evaluate loss decrease (`delta_loss`) and relative decrease (`relative_gain`)
  in parallel, with no primary/secondary ranking. The user did not designate
  either as primary. Present both in notebook summaries, tables, and conclusions.
- Split by source/problem groups and check exact and near duplicates across
  A1, A2, and B. Preserve dataset revisions, IDs, split seeds, and manifests.
- Evaluate reference and fine-tuned models on identical tokenized examples,
  attention masks, and response masks. Report per-example mean response-token
  negative log likelihood; do not substitute a batch-averaged training loss.
- Positive `loss_ref - loss_sft` means the fine-tuned model improved on an example.
  Use A2 as the positive class and do not reverse score directions after seeing AUC.
- Include raw-loss baselines and length/format controls. Label proposed results,
  observed results, and interpretations clearly; never invent experimental results.
- Use float32 forward passes for scoring: the bf16 pilot sanity check showed
  batch/padding-dependent mean NLL differences around 0.0047. Training uses
  float32 master parameters and optimizer state with bf16 autocast.
- On the current local PyTorch/CUDA stack, training explicitly uses the SDPA
  math backend through both forward and backward: automatic backend selection
  produced non-finite bf16 gradients on a real batch, with or without activation
  checkpointing. Do not remove the finite-gradient check to bypass this failure.
- The Dolly extension is configured in `configs/dolly_extension.yaml`: A2/B/C
  each contain 512 fixed examples and reuse the original reference and step-32
  SFT models. It is evaluation only, with no optimizer updates.
- C is a uniform seeded sample of Dolly `general_qa`, filtered only for empty or
  unscorable content and full-template length above 2,048 tokens. Do not truncate,
  impose a minimum answer length, filter topics/languages, or manually replace rows.
  Retain duplicate questions and different answers; record normalized question
  groups for bootstrap intervals and report overlaps with A1/A2/B separately.
- Gradient diagnostics use per-example mean response NLL, full float32, eval
  mode and SDPA math. Measure unclipped gradients and count tied parameters once.
  The directional reference uses only a fixed A1 sample, never A2/B/C.
- The review notebooks preserve their loss sections and append gradient
  diagnostics for the same two-domain comparison. Dolly contains A2 versus
  Dolly, without B comparisons; its length control uses only A2/C. The A2/B
  appendix may reuse the verified A2/B measurements from the Dolly run.
  Preserve existing cells and review notes when adding or replacing an appendix.
- The UltraChat replication uses `configs/ultrachat_extension.yaml`: fixed
  `HuggingFaceH4/ultrachat_200k` `test_sft`, first complete user/assistant pair
  per conversation, length/basic-validity filtering only, uniform 512-row sample.
  Measure and retain loss plus gradient diagnostics for A2/C on both models;
  its notebook follows Dolly's loss presentation with a gradient appendix. Document upstream data
  processing and distinguish a source replication from a multiple-seed estimate.

# Artifact storage

- Store heavyweight project training and data artifacts under
  `/mnt/tianzhu_hot/data-picking`: downloaded datasets and model/tokenizer caches,
  processed data, manifests, checkpoints, logs, scores, metrics, and figures.
  Keep source code, checked-in configuration, and documentation in this repository.
- Store reviewable experiment analyses in the repository under
  `experiments-analysis/YYYY-MM-DD_HHMMSSZ_<run_id>/analysis.ipynb`, ordered by
  result time in UTC. Keep executed notebook outputs and small, checksummed
  result/configuration snapshots there so reviews do not require the mount.
  This is the user-requested exception to the external artifact location.
  Keep checkpoints, model weights, downloaded datasets and token caches on the
  mount; record their original paths in each analysis manifest.
- Keep an experiment index in `experiments-analysis/README.md`. Preserve the
  directory timestamp on later edits, document its source, and do not silently
  overwrite an analysis notebook that may contain manual review notes.
- Suggested layout: `cache/`, `data/raw/`, `data/processed/`, and
  `runs/<run_id>/{checkpoints,logs,scores,metrics,figures}` under that root.
- Explicitly configure library cache and trainer output paths; do not silently
  place experiment artifacts in the repository, a home-directory cache, or `/tmp`.
- The root is currently a writable blobfuse2 mount. Check mount availability and
  writability before training. Checkpoint completion must use a manifest and a
  completion marker written after all checkpoint files have been closed and
  validated; do not rely on atomic directory rename on object-backed storage.
- Transient POSIX scratch files (including dependency installation scratch) may
  use `/dev/shm/data-picking-*`: blobfuse2 cannot unlink open temporary files.
  This is not a storage location for retained experiment artifacts.
- Large mounted checkpoints may need a temporary RAM copy for evaluation.
  `scripts/stage_evaluation_weights.py` verifies original SHA-256 values before
  use; persistent sources stay on the mount. Avoid CUDA copies directly from
  mmap-backed blobfuse tensors. Never log the blob mount's credentials or signed URLs.

# Training experiment preferences

- When creating a new training experiment, verify that checkpoints save optimizer
  state and automatic resume is enabled, unless the user explicitly requests otherwise.
- In `/home/v-guanhechen/rl`, experiments using
  `scripts/train-sopd-sing/train.sh` must explicitly set `--save_optimizer True`
  and `--auto_resume True`, with a valid persistent `--autosync_dir`. Check the
  effective configuration before launching or handing off the experiment.
- For a new trainer, also restore scheduler, RNG, global step, and data iteration
  state. Reject automatic resume when the model, data, or training configuration
  does not match the saved run.
