#!/usr/bin/env python3
"""Fresh cumulative fits with identical train, validation and test transcripts.

The complete cohort is fixed over all 114 datasets before any subset is trained.
Existing legacy results are never modified or resumed by this entrypoint.
"""
from run_cumulative_stability import main

if __name__ == '__main__':
    raise SystemExit(main(fixed_cohort=True))
