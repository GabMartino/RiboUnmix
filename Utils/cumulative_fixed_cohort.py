"""Fixed, complete transcript cohorts for a controlled cumulative experiment."""
from __future__ import annotations

import numpy as np

from Utils.real_panel_convergence import load_sequence_metadata, load_support_and_stored_weights, utc_timestamp
from Utils.reliability_references import transcript_id_hash
from Utils.stratified_transcript_split import assign_reliability_quantile_bins, sample_stratified_ids


def partition_complete_cohort(metadata, supports, stored_weights, *, seed, validation_fraction=.1, test_fraction=.1, reliability_bins=10):
    """Assign each transcript once, using the intersection across every dataset."""
    complete = sorted(set(metadata.index).intersection(*supports.values()))
    test_count = int(round(len(complete) * test_fraction))
    validation_count = int(round(len(complete) * validation_fraction))
    if min(test_count, validation_count, len(complete) - test_count - validation_count) < 1:
        raise ValueError('The complete observation intersection is too small for three nonempty folds.')
    reliability = {tid: float(np.median([stored_weights[d][tid] for d in supports])) for tid in complete}
    bins = assign_reliability_quantile_bins(reliability, number_of_bins=reliability_bins)
    strata = {tid: f'qbin_{bins[tid]:02d}__{metadata.at[tid, "css_bin"]}' for tid in complete}
    rng = np.random.default_rng(seed)
    test = sample_stratified_ids(candidate_ids=complete, stratum_by_transcript=strata, target_count=test_count, rng=rng)
    remaining = sorted(set(complete) - set(test))
    validation = sample_stratified_ids(candidate_ids=remaining, stratum_by_transcript=strata, target_count=validation_count, rng=rng)
    train = sorted(set(remaining) - set(validation))
    return dict(train=sorted(train), validation=sorted(validation), test=sorted(test), complete=complete)


def build_fixed_cumulative_split(*, experiment_name, tasks, dataset_mapping, sequences_path, subset_seed,
                                 validation_fraction, test_fraction, reliability_bins, maximum_cds_codons):
    metadata, sequence_report = load_sequence_metadata(sequences_path, max_cds_codons=maximum_cds_codons)
    supports, weights, eligibility = load_support_and_stored_weights(dataset_mapping, eligible_sequence_ids=set(metadata.index))
    folds = partition_complete_cohort(metadata, supports, weights, seed=subset_seed,
        validation_fraction=validation_fraction, test_fraction=test_fraction, reliability_bins=reliability_bins)
    panels = {t['run_id']: list(t['datasets']) for t in tasks}
    return dict(manifest_version=1, experiment_name=experiment_name, created_at_utc=utc_timestamp(), random_seed=subset_seed,
        source_sequences_path=str(sequences_path), maximum_cds_codons=maximum_cds_codons,
        sequence_eligibility_report=sequence_report, dataset_pair_eligibility_reports=eligibility,
        split_protocol='fixed_complete_transcripts', complete_observation_datasets=list(dataset_mapping),
        complete_transcript_ids=folds['complete'], complete_transcript_count=len(folds['complete']),
        cohort_limitation='Requires usable observations in every configured dataset; favors broadly observed transcripts.',
        panels=panels, panel_train_eligible_ids={p: folds['train'] for p in panels},
        panel_validation_ids={p: folds['validation'] for p in panels},
        common_training_ids=folds['train'], common_validation_ids=folds['validation'], common_test_ids=folds['test'],
        common_test_source='complete_observation_intersection_stratified', common_test_requires_support_in_every_subset=True,
        common_test_prediction_mode='sequence_only_shared_profile',
        minimum_usable_datasets_per_training_or_validation_transcript=2,
        panel_support_statistics={p: dict(number_of_train_eligible_transcripts=len(folds['train']),
            number_of_validation_transcripts=len(folds['validation']), number_of_common_test_sequences=len(folds['test']),
            training_support_minimum=len(names), training_support_median=len(names), training_support_maximum=len(names),
            common_test_observed_in_selected_subset=len(folds['test'])) for p, names in panels.items()},
        fold_id_hashes=dict(train=transcript_id_hash(folds['train']), validation=transcript_id_hash(folds['validation']),
            test=transcript_id_hash(folds['test']), validation_by_panel={p: transcript_id_hash(folds['validation']) for p in panels}),
        assertions=dict(train_validation_test_disjoint=True, identical_training_ids_across_all_sizes=True,
            identical_validation_ids_across_all_sizes=True, identical_test_ids_across_all_sizes=True,
            every_fold_transcript_observed_in_every_dataset=True, no_transcript_changes_fold_between_sizes=True))
