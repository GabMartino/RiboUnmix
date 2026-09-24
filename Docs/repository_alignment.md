# Public repository scope

This repository was assembled as a reviewer-facing source release on
24 September 2026. Canonical RiboUnmix modules, compatibility adapters,
configuration files, preprocessing source, analysis source, tests, and a small
end-to-end synthetic fixture are committed together.

## Included

- `Models/RiboUnmixModel/` and `Models/RiboUnmixLightningModule.py`;
- canonical multidataset, synthetic, and benchmarking entrypoints;
- canonical loaders under `Dataloaders/RiboUnmix*/`;
- frozen configuration and experiment-design source;
- numerical, loss, reference-centering, split, and batching tests;
- analysis scripts, excluding generated `analyses/artifacts/` outputs;
- documentation and model diagrams;
- the CPU-runnable `examples/synthetic_smoke/` integration example;
- legacy import adapters needed to load historical checkpoints.

## Intentionally excluded

- every Slurm batch script and IDE metadata;
- `Datasets/data/` and other complete real/synthetic data collections;
- full experiment outputs, logs, figures, and manuscript build products;
- machine-specific virtual environments and scheduler logs.

The small public fixture is not a statistically meaningful replacement for the
complete data. Its data and checkpoint manifests contain SHA-256 hashes and
make its limited role explicit.

## Scientific cautions

1. **Shared does not mean biologically identified.** The fixed reference
   defines a decomposition convention; it does not prove that all technical
   effects are confined to gamma.
2. **Mass conservation is experiment-specific.** Use the resolved run config,
   rather than a current default, to describe a historical result.
3. **Validation is not independent testing.** The public smoke checkpoint is
   selected and scored on its validation fold and exists only as an integration
   fixture.
4. **Compatibility names preserve provenance.** Earlier `RiboAI`, `Queueing`,
   and `Queuing` paths remain as thin adapters so old checkpoints and commands
   are not silently broken.

## Minimal verification

```bash
python examples/synthetic_smoke/run.py verify
python examples/synthetic_smoke/run.py replay
python Tests/run_public_tests.py
```

This public profile excludes integration tests whose authoritative inputs live
in the omitted full data or historical results trees.

The project still needs an author-selected software license and finalized
citation metadata. Those are policy decisions and were not inferred during the
technical release cleanup.
