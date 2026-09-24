# N114 bias-GRU precision repair — 2026-09-09

## Evidence and limits

The latest Leonardo traceback is **after backward, before clipping**, at epoch
9, execution batch 3966, for `ENST00000426508.7` (1463 modeled codons, including
the terminal stop). Eight tensors are affected: the four bias-branch embeddings
and the forward-direction layer-0 GRU weights/biases. The biological encoder,
observation head, and later recurrent layers are not reported as affected.

This localizes the failing parameter gradients to the early bias recurrent
path. It does not by itself identify a particular CUDA kernel. Increasing host
RAM, lowering DataLoader worker counts, bounding gamma again, or clipping after
backward cannot repair an already non-finite recurrent derivative.

The local checkpoint is older: epoch 3, global step 1668. This workstation has
no usable CUDA device. The exact epoch-9 CUDA failure therefore has **not** been
replayed, and a successful complete N114 continuation is not claimed.

## Implemented repair

The new optional execution setting is:

```text
++model.dataset_bias_params.context_gru_precision=float32
```

It keeps the trainable dataset/codon/nucleotide/amino-acid embeddings in FP32
before assembling the recurrent input, and disables autocast around the bias
GRU and its output LayerNorm. Merely casting a BF16 recurrent output to FP32
would be too late. Embeddings are also not cast to BF16 and back on their way
into the GRU. The implementation uses the autocast-disabled subregion approach
described in the [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html).

The biological branch and observation/alpha heads retain the configured Trainer
precision (`bf16-mixed` in this run). Fixed-reference evaluation uses the same
repaired branch for **every** selected reference identity, keeping autograd
attached. All finite-gradient guards remain enabled; no gradients or training
rows are dropped, sanitized, or silently retried.

Unchanged: architecture, parameter names/order/shapes, gamma bounds and gauges,
uniform reference weights, reliability weights, loss terms, optimizer, logical
batch sizes, accumulation, reference chunk sizes, and transcript splits.

The base YAML is deliberately unchanged. Missing precision settings mean
`inherit`, preserving other launches and their resolved configuration hashes.
The N114 resume wrapper opts in explicitly. An explicit resume precision choice
is saved in `resume_manifest.json` and retained on later submissions even if the
CLI option is omitted. `--bias-gru-precision inherit` explicitly restores the old
execution choice. No ranked-reference launcher or completed prediction was edited.

Full-state resume restores the saved model, Adam, scheduler, and Trainer state.
Changing arithmetic precision preserves the logical objective, **not** a
bitwise-identical future trajectory. Resume starts at the latest usable saved
checkpoint, not at an unsaved failing backward pass.

## Verification

60 focused unittest cases passed, as did five additional execution-microbatch
checks and a Hydra configuration/run-tag check. Python compilation, Slurm shell
syntax, and `git diff --check` passed. `pytest` is not installed in this local
venv; unittest cases ran with `python -m unittest`, and the five plain
microbatch test functions were invoked directly.

- A deterministic adversarial 1463-step GRU test has finite forward outputs
  but infinite bias gradients under BF16 autocast. With identical parameters
  and inputs, the FP32 island gives finite gradients. This tests the numerical
  mechanism, not the unavailable real epoch-9 state.
- FP32-island recurrent outputs and gradients match explicit FP32 execution.
- Hooks verify FP32 recurrent inputs with autocast disabled; observation heads
  still execute under BF16 autocast. Tests also reject a cast-to-BF16-then-back
  embedding path.
- Old state dictionaries load strictly, with unchanged parameter keys/order;
  Adam step counters and moment tensors restore unchanged.
- The actual fixed-reference centering function agrees across reference chunk
  sizes 1 and 4 on a small dropout-free test, including all reference gradients.
- The existing loss, gamma-range, alpha-gradient-routing, finite-gradient guard,
  and logical-microbatch invariance tests are retained.
- A temporary-tree N114 dry run restores full state, leaves N40 untouched, and
  retains standard NB, BF16 Trainer precision, equal pi, and logical batch size.
- Hydra composition changes only the new precision key and preserves the
  existing checkpoint run tag. Compilation and shell syntax checks pass.

A read-only, bounded-memory probe also loaded the real epoch-3 bias weights and
the reported 1463-codon sequence. BF16 checks over the selected identities and
FP32-island checks of the first eight identities were finite. These used a squared-raw-score surrogate to
exercise the recurrent derivative, not the complete measured NB/PCC objective;
they must not be presented as a replay of the failing training update. The
FP32-island probe peaked below 1 GiB RSS for four identities at a time.

## Resume only N114 on Leonardo

Copy the updated files to the cluster first, **not only the Slurm wrapper**:

- `Models/RiboUnmixModel/DatasetBiasSubmodel.py`
- `main_ribounmix_multidataset.py`
- `resume_real_experiment_from_checkpoints.py`
- `resume_real_exp8_N114_256G.slurm`

From the project directory:

```bash
sbatch --export=ALL,RUN_ROOT=/leonardo_work/EUHPC_D35_089/my_exp8_a100_b32_20260906_114340,BIAS_GRU_PRECISION=float32 \
  resume_real_exp8_N114_256G.slurm
```

The wrapper requests one GPU, 256 GiB host RAM, the long production QoS, zero
training DataLoader workers, and the unchanged execution/batch limits. It selects
only N114 and enables the FP32 bias context by default. It does not warm-start a
weights-only checkpoint without explicit consent. Expect these log messages:

```text
Resume launcher version: 2026-09-09.bias-gru-fp32-v5
Bias context GRU precision: float32; Trainer precision: bf16-mixed.
Resuming complete Trainer state (model, optimizer, scheduler, epoch, callbacks, and loops).
```

FP32 can use more GPU activation memory and may cost throughput; 256 GiB host
RAM does not increase GPU VRAM. If a further numerical failure occurs, retain
the new full log and checkpoint. Setting `DETECT_ANOMALY=1` on this same wrapper
adds diagnostic tracing without turning off the repair or the fail-fast guard.
Do not claim that FP32 makes every possible recurrent gradient finite.
