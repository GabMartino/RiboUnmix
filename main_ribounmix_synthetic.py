"""Entrypoint for the synthetic bias-recovery sanity check.

Reuses the full multi-dataset training/validation-split/prediction pipeline
in main_ribounmix_multidataset.py unchanged, pointed at the
synthetic dataset config/encoding via config_ribounmix_synthetic.yaml.
"""

import os

import hydra

from main_ribounmix_multidataset import main as _main

# hydra.main() infers config_path relative to the caller by reading
# task_function.__module__; since _main.__wrapped__ was defined in a
# different file, that resolves to that module's name (not "__main__") and
# Hydra instead treats config_path as a python package. An absolute path
# bypasses that detection entirely.
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")

main = hydra.main(
    version_base=None,
    config_path=_CONFIG_DIR,
    config_name="config_ribounmix_synthetic",
)(_main.__wrapped__)

if __name__ == "__main__":
    main()
