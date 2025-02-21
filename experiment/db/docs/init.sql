CREATE TABLE EXPERIMENT (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(255),
    desc TEXT,
    start_time DATETIME,
    update_time DATETIME,
    config TEXT,
    dataset_id VARCHAR(255), 
    FOREIGN KEY (dataset_id) REFERENCES DATASET(id)
);

CREATE TABLE TRIAL (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255),
    desc TEXT,
    start_time DATETIME,
    update_time DATETIME,
    config TEXT,
    experiment_id VARCHAR(255), 
    FOREIGN KEY (experiment_id) REFERENCES EXPERIMENT(id)
);

CREATE TABLE TRIAL_RUN (
    id VARCHAR(255) PRIMARY KEY,
    start_time DATETIME,
    end_time DATETIME,
    status VARCHAR(255),
    trial_id VARCHAR(255),
    FOREIGN KEY (trial_id) REFERENCES TRIAL(id)
);

CREATE TABLE RESULTS (
    id VARCHAR(255) PRIMARY KEY,
    trial_run_id VARCHAR(255),
    total_accuracy FLOAT,
    accuracy_per_label TEXT,
    total_loss FLOAT,
    loss_per_label TEXT,
    FOREIGN KEY (trial_run_id) REFERENCES TRIAL_RUN(id)
);

CREATE TABLE EPOCH (
    id VARCHAR(255) PRIMARY KEY,
    trial_run_id VARCHAR(255),
    index INT,
    total_accuracy FLOAT,
    accuracy_per_label TEXT,
    total_loss FLOAT,
    loss_per_label TEXT,
    FOREIGN KEY (trial_run_id) REFERENCES TRIAL_RUN(id)
);

CREATE TABLE DATASET (
    id VARCHAR(255) PRIMARY KEY,
    size FLOAT,
    location VARCHAR(255),
    config TEXT
);

CREATE TABLE ENCODER (
    type VARCHAR(255) PRIMARY KEY,
    config TEXT
);

CREATE TABLE ARTIFACT (
    id VARCHAR(255) PRIMARY KEY,
    type VARCHAR(255),
    config TEXT,
    location VARCHAR(255),
    trial_run_id VARCHAR(255),
    epoch_id VARCHAR(255),
    results_id VARCHAR(255),
    FOREIGN KEY (trial_run_id) REFERENCES TRIAL_RUN(id),
    FOREIGN KEY (epoch_id) REFERENCES EPOCH(id),
    FOREIGN KEY (results_id) REFERENCES RESULTS(id)
);

CREATE TABLE LOGS (
    id VARCHAR(255) PRIMARY KEY,
    location VARCHAR(255),
    trial_run_id VARCHAR(255),
    FOREIGN KEY (trial_run_id) REFERENCES TRIAL_RUN(id)
);

CREATE TABLE EXPERIMENT_DATASET (
    experiment_id VARCHAR(255),
    dataset_id VARCHAR(255),
    PRIMARY KEY (experiment_id, dataset_id),
    FOREIGN KEY (experiment_id) REFERENCES EXPERIMENT(id),
    FOREIGN KEY (dataset_id) REFERENCES DATASET(id)
);

CREATE TABLE ENCODER_DATASET (
    encoder_type VARCHAR(255),
    dataset_id VARCHAR(255),
    PRIMARY KEY (encoder_type, dataset_id),
    FOREIGN KEY (encoder_type) REFERENCES ENCODER(type),
    FOREIGN KEY (dataset_id) REFERENCES DATASET(id)
);
