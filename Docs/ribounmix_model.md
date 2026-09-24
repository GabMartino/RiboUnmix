# RiboUnmix model documentation

The current executable data flow, weighting, split, grouped batching,
joint fixed-reference/positional gamma gauges, replica objective, and selectable
sample reduction (transcript-balanced by default) are
documented in [`model_mathematics.html`](model_mathematics.html).

The earlier configuration summary is retained as a
[historical snapshot](current_configured_model.md). For current defaults, use
`python main_ribounmix_multidataset.py --cfg job --resolve`; the
[README](../README.md) explains the differences between experiment families.

Older content in this file described removed additive/gating branches,
consensus-loss modes, and legacy selectors. It was deleted so that obsolete
equations and configuration names cannot be mistaken for the current model.
