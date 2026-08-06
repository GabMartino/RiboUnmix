




## Conserved stalling sites

- stalling_sites.info.json
  - {"num_transcripts": 1729, "num_total_conserved_stalling_sites": 2426}

- stalling_sites.parquet: [1729 rows x 2 columns]
  - columns = ["transcript_id", "conserved_stalling_sites"]
  - "transcript_id": ENST00000000233
  - "conserved_stalling_sites": [16, 127, 68]


## Data


### raw_datasets_ribo_only_with_css to filtered_datasets_ribo_only_with_css
This datasets contain the transcripts for each dataset filtered by: 
mask = (log_density >= 0.5) | (coverage >= 0.1)
filter_datasets.py from raw_datasets_ribo_only_with_css

### raw_datasets_with_css to raw_datasets_ribo_only_with_css
strip_dataset_from_sequence.py
extract from raw_datasets only ["id", "transcript_id", "ribo"]

### raw_datasets



- sequence:
  - css_split.json (legacy artifact; the active multidataset train/validation split does not read it):
    - training_set: ["ENST00000355849.10", "ENST00000361794.7",
    - validation_set: ["ENST00000355849.10", "ENST00000361794.7",
- raw_full_dataset_replicas

- raw_datasets:
