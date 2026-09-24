# RiboUnmix naming migration

The project is named **RiboUnmix**. The canonical Python API, training
entrypoints, and Hydra configurations use the `RiboUnmix` name.

The earlier `RiboAI`, `Queueing`, and `Queuing` module paths remain only as
deprecated compatibility adapters. They are required to load frozen experiment
commands and checkpoints whose provenance records those import paths. New code
must not import the compatibility modules.

Historical result directories, run identifiers, manifests, and checkpoint
metadata are intentionally not rewritten. They are immutable scientific
provenance, and changing their text would invalidate recorded hashes and make
exact resumption harder to audit.

The local checkout directory may also retain its earlier filesystem name while
jobs are active. The repository URL is:

```text
https://github.com/GabMartino/RiboUnmix.git
```

Canonical entrypoints:

- `main_ribounmix_multidataset.py`
- `main_ribounmix_benchmarking.py`
- `main_ribounmix_synthetic.py`

Canonical configurations:

- `config/config_ribounmix_multidataset.yaml`
- `config/config_ribounmix_benchmarking.yaml`
- `config/config_ribounmix_synthetic.yaml`
- `config/config_ribounmix_synthetic_alpha_causality.yaml`

Canonical environment variables:

- `RIBOUNMIX_LOGGER_VERSION`
- `RIBOUNMIX_PLOT_TEX`

For a transition period, the corresponding `RIBOAI_*` variables are accepted
as fallbacks. If both are set, the canonical `RIBOUNMIX_*` value wins.

## Deployment set

Deploy the following canonical implementation paths together:

```text
Dataloaders/RiboUnmixBenchmarking/
Dataloaders/RiboUnmixMultiDataset/
Models/RiboUnmixModel/
Models/RiboUnmixLightningModule.py
main_ribounmix_benchmarking.py
main_ribounmix_multidataset.py
main_ribounmix_synthetic.py
config/config_ribounmix_benchmarking.yaml
config/config_ribounmix_multidataset.yaml
config/config_ribounmix_synthetic.yaml
config/config_ribounmix_synthetic_alpha_causality.yaml
config/model/ribounmix_nb.yaml
Utils/publication_plot_style.py
```

For compatibility with already frozen commands and checkpoints, also deploy:

```text
Dataloaders/RiboAIQueueingBenchmarking/
Dataloaders/RiboAIQueuingMultiDataset/
Models/RiboQueuingModel/
Models/RiboQueuingModelLighningModule.py
main_ribo_queueing_modeling_benchmarking.py
main_ribo_queueing_modeling_multi_dataset.py
main_ribo_queueing_modeling_synthetic.py
config/config_riboai_queueing_benchmarking.yaml
config/config_riboai_queuing_multidataset.yaml
config/config_riboai_queueing_synthetic.yaml
config/config_riboai_queueing_synthetic_alpha_causality.yaml
config/model/ribo_queue_simple_nb.yaml
```

Launchers, analysis source, tests, and maintained documentation now refer to
the canonical names. Deploying by Git commit is therefore safer than copying
only one entrypoint. Historical `results/`, `analyses/artifacts/`, checkpoint,
and log trees must remain unchanged.
