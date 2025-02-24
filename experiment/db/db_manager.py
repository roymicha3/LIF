import os
from datetime import datetime
from sqlalchemy import create_engine
from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker

from experiment.db.tables import Base, Experiment, Trial, TrialRun, Results, Epoch, Metric, Artifact

DB_URL_PREFIX = "sqlite:///"

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.db_url = f"{DB_URL_PREFIX}{self.db_path}"
        self.engine = create_engine(self.db_url)
        
        self.data_path = os.path.join(self.db_dir_path, "data")
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
        self.Session = sessionmaker(bind=self.engine)

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def drop_tables(self):
        Base.metadata.drop_all(self.engine)

    # Experiment methods
    def create_experiment(self, title, desc):
        with self.session_scope() as session:
            experiment = Experiment(title=title, desc=desc, start_time=datetime.now(), update_time=datetime.now())
            session.add(experiment)
            session.flush()
            return experiment.id

    def get_experiment(self, experiment_id):
        with self.session_scope() as session:
            return session.query(Experiment).get(experiment_id)

    def update_experiment(self, experiment_id, **kwargs):
        with self.session_scope() as session:
            experiment = session.query(Experiment).get(experiment_id)
            for key, value in kwargs.items():
                setattr(experiment, key, value)
            experiment.update_time = datetime.now()

    def add_artifact_to_experiment(self, experiment_id, artifact_id):
        with self.session_scope() as session:
            experiment = session.query(Experiment).get(experiment_id)
            artifact = session.query(Artifact).get(artifact_id)
            experiment.artifacts.append(artifact)

    # Trial methods
    def create_trial(self, experiment_id, name):
        with self.session_scope() as session:
            trial = Trial(experiment_id=experiment_id, name=name, start_time=datetime.now(), update_time=datetime.now())
            session.add(trial)
            session.flush()
            return trial.id

    def get_trial(self, trial_id):
        with self.session_scope() as session:
            return session.query(Trial).get(trial_id)

    def add_artifact_to_trial(self, trial_id, artifact_id):
        with self.session_scope() as session:
            trial = session.query(Trial).get(trial_id)
            artifact = session.query(Artifact).get(artifact_id)
            trial.artifacts.append(artifact)

    # TrialRun methods
    def create_trial_run(self, trial_id, status):
        with self.session_scope() as session:
            trial_run = TrialRun(trial_id=trial_id, status=status, start_time=datetime.now(), update_time=datetime.now())
            session.add(trial_run)
            session.flush()
            return trial_run.id

    def update_trial_run_status(self, trial_run_id, status):
        with self.session_scope() as session:
            trial_run = session.query(TrialRun).get(trial_run_id)
            trial_run.status = status
            trial_run.update_time = datetime.now()

    def add_artifact_to_trial_run(self, trial_run_id, artifact_id):
        with self.session_scope() as session:
            trial_run = session.query(TrialRun).get(trial_run_id)
            artifact = session.query(Artifact).get(artifact_id)
            trial_run.artifacts.append(artifact)

    # Results methods
    def create_results(self, trial_run_id):
        with self.session_scope() as session:
            results = Results(trial_run_id=trial_run_id, time=datetime.now())
            session.add(results)
            session.flush()
            return results.trial_run_id

    # Epoch methods
    def create_epoch(self, trial_run_id, idx):
        with self.session_scope() as session:
            epoch = Epoch(trial_run_id=trial_run_id, idx=idx, time=datetime.now())
            session.add(epoch)
            session.flush()
            return epoch.idx, epoch.trial_run_id

    # Metric methods
    def create_metric(self, type, total_val, per_label_val=None):
        with self.session_scope() as session:
            metric = Metric(type=type, total_val=total_val, per_label_val=per_label_val)
            session.add(metric)
            session.flush()
            return metric.id

    def add_metric_to_results(self, results_id, metric_id):
        with self.session_scope() as session:
            results = session.query(Results).get(results_id)
            metric = session.query(Metric).get(metric_id)
            results.metrics.append(metric)

    def add_metric_to_epoch(self, epoch_idx, epoch_trial_run_id, metric_id):
        with self.session_scope() as session:
            epoch = session.query(Epoch).filter_by(idx=epoch_idx, trial_run_id=epoch_trial_run_id).first()
            metric = session.query(Metric).get(metric_id)
            epoch.metrics.append(metric)

    # Artifact methods
    def create_artifact(self, type, loc):
        with self.session_scope() as session:
            artifact = Artifact(type=type, loc=loc)
            session.add(artifact)
            session.flush()
            return artifact.id

    def add_artifact_to_results(self, results_id, artifact_id):
        with self.session_scope() as session:
            results = session.query(Results).get(results_id)
            artifact = session.query(Artifact).get(artifact_id)
            results.artifacts.append(artifact)

    def add_artifact_to_epoch(self, epoch_idx, epoch_trial_run_id, artifact_id):
        with self.session_scope() as session:
            epoch = session.query(Epoch).filter_by(idx=epoch_idx, trial_run_id=epoch_trial_run_id).first()
            artifact = session.query(Artifact).get(artifact_id)
            epoch.artifacts.append(artifact)

    @property
    def db_dir_path(self):
        return os.path.dirname(self.db_path)
