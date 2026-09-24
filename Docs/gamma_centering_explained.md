# Gamma centering: reader guide

The current, source-aligned explanation is the **Fixed-reference gamma
centering** section of [`model_mathematics.html`](model_mathematics.html). The
compact reference equations and configuration are also available in
[`gamma_centering.md`](gamma_centering.md).

The core idea is:

1. the model predicts a raw log correction for a requested
   transcript–dataset pair;
2. it internally evaluates the same transcript against the checkpointed fixed
   reference-dataset panel;
3. it subtracts the quality-rank-weighted mean reference log correction at
   each valid position;
4. exponentiation produces a positive gamma whose weighted geometric mean is
   one over the fixed panel.

This gauge is independent of the physical batch. A singleton inference request
is therefore still centered using the panel saved in the checkpoint. Dataset
quality weights used for this gauge are distinct from transcript reliability
weights used by the loss.

Earlier versions of this file described a removed sparse support gate,
additive branch, and batch-local equal-dataset center. Those descriptions were
deleted because they no longer match the executable model.
