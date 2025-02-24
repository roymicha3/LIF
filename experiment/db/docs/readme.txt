- Tables:

* Experiment:
    id = PK
    title
    desc
    start_time
    update_time

* Trial:
    id = PK
    experiment_id = FK
    name
    start_time
    update_time

* Trial Run:
    id = PK
    trial_id = FK
    status
    start_time
    update_time

* Results:
    trial_run_id = PK, FK
    time

* Epoch:
    idx = CK
    trial_run_id = CK
    time

* Metric:
    id = PK
    type
    total_val
    per_label_val, ...

* Artifact:
    id = PK
    type
    loc

- Relations:

* Trial Run - Artifact:
    artifact_id = FK, PK
    trial_run_id = FK

* Results - Metric:
    metric_id = FK, PK
    trial_run_id = FK

* Results - Artifact:
    artifact_id = FK, PK
    trial_run_id = FK

* Epoch - Metric:
    metric_id = FK, PK
    trial_run_id = FK
    epoch_idx = FK

* Epoch - Artifact:
    artifact_id = FK, PK
    trial_run_id = FK
    epoch_idx = FK