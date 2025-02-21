# Experiment Tracking Database Schema

This database schema is designed to support an **experiment tracking system** that records trials, trial runs, results, artifacts, datasets, encoders, and logs.

The data base is desigend as:
    1. DB file
    2. data - directory containing resources and artifacts

## 📌 Table Overview

### 1️⃣ `EXPERIMENT`

Represents an experiment consisting of multiple trials.

- `id` (PK) - Unique identifier for the experiment.
- `title` - Name of the experiment.
- `desc` - Description of the experiment.
- `start_time` - When the experiment started.
- `update_time` - Last update timestamp.
- `config` - Configuration details (e.g., JSON format).
- `dataset_id` (FK) - Links the experiment to a dataset.

### 2️⃣ `TRIAL`

Represents a specific trial within an experiment.

- `id` (PK) - Unique identifier for the trial.
- `name` - Name of the trial.
- `desc` - Description of the trial.
- `start_time` - When the trial started.
- `update_time` - Last update timestamp.
- `config` - Trial-specific configurations.
- `experiment_id` (FK) - Links the trial to its parent experiment.

### 3️⃣ `TRIAL_RUN`

Each trial consists of multiple trial runs.

- `id` (PK) - Unique identifier for the trial run.
- `start_time` - When the trial run started.
- `end_time` - When the trial run ended.
- `status` - Status of the trial run (e.g., completed, failed).
- `trial_id` (FK) - Links the trial run to its parent trial.

### 4️⃣ `RESULTS`

Stores the results of a trial run.

- `id` (PK) - Unique identifier for the result entry.
- `trial_run_id` (FK) - Links results to a specific trial run.
- `total_accuracy` - Overall accuracy of the model.
- `accuracy_per_label` - Accuracy per label (JSON format).
- `total_loss` - Overall loss value.
- `loss_per_label` - Loss per label (JSON format).

### 5️⃣ `EPOCH`

Records performance per training epoch within a trial run.

- `id` (PK) - Unique identifier for the epoch.
- `trial_run_id` (FK) - Links epoch to a trial run.
- `index` - Epoch number.
- `total_accuracy` - Accuracy after the epoch.
- `accuracy_per_label` - Accuracy breakdown.
- `total_loss` - Loss after the epoch.
- `loss_per_label` - Loss breakdown.

### 6️⃣ `DATASET`

Represents datasets used in experiments.

- `id` (PK) - Unique identifier for the dataset.
- `size` - Size of the dataset.
- `location` - File location or source.
- `config` - Dataset-specific configurations.

### 7️⃣ `ENCODER`

Defines encoding methods used in data processing.

- `type` (PK) - Type of encoder (e.g., BERT, One-Hot).
- `config` - Encoder-specific configurations.

### 8️⃣ `ARTIFACT`

Represents artifacts generated during a trial run.

- `id` (PK) - Unique identifier for the artifact.
- `type` - Type of artifact (e.g., model checkpoint, log file).
- `config` - Configuration details.
- `location` - Storage location.
- `trial_run_id` (FK) - Links to a trial run.
- `epoch_id` (FK) - Links to a specific epoch.
- `results_id` (FK) - Links to result data.

### 9️⃣ `LOGS`

Stores log files related to trial runs.

- `id` (PK) - Unique identifier for the log entry.
- `location` - Storage location of logs.
- `trial_run_id` (FK) - Links logs to a trial run.

### 🔁 Relationship Tables

#### `EXPERIMENT_DATASET`

Links experiments to datasets (many-to-many relationship).

- `experiment_id` (FK) - Experiment reference.
- `dataset_id` (FK) - Dataset reference.

#### `ENCODER_DATASET`

Links encoders to datasets (many-to-many relationship).

- `encoder_type` (FK) - Encoder reference.
- `dataset_id` (FK) - Dataset reference.

## 🔍 How It Works

1. **An **``** is created**, linked to one or more `DATASET`s.
2. **Each experiment consists of multiple **``**s**, which explore different configurations.
3. **A **``** has multiple **``**s**, each representing a single execution.
4. **During each **``**, **``**, **``**s, **``**, and **``**s are recorded**.
5. **Encoders and datasets are managed separately and linked to experiments as needed**.

## 📂 Example Queries

### 1️⃣ Get all trials for a given experiment

```sql
SELECT * FROM TRIAL WHERE experiment_id = 'exp_123';
```

### 2️⃣ Retrieve trial run results

```sql
SELECT * FROM RESULTS WHERE trial_run_id = 'run_456';
```

### 3️⃣ Find all artifacts related to a trial run

```sql
SELECT * FROM ARTIFACT WHERE trial_run_id = 'run_789';
```

---

This database schema provides a **structured and scalable approach** for tracking experiments, trial runs, and model performance data. 🚀

