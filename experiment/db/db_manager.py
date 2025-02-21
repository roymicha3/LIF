import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import json

from experiment.db.tables import *


URI = "sqlite:///"
DATA_DIR = "data"

Base = declarative_base()

class DBManager:
    def __init__(self, db_path):
        """
        Initialize the database manager.
        """
        db_url = f"{URI}{db_path}"
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        base_dir = os.path.dirname(db_path)
        self.data_path = os.path.join(base_dir, DATA_DIR)
        os.makedirs(self.data_path, exist_ok=True)

    def add_experiment(self, title, desc, config, dataset_id):
        """
        Add a new experiment to the database.
        """
        session = self.Session()
        experiment = Experiment(
            id=generate_id(),
            title=title,
            desc=desc,
            start_time=datetime.now(),
            update_time=datetime.now(),
            config=json.dumps(config),
            dataset_id=dataset_id
        )
        session.add(experiment)
        session.commit()
        session.close()
        return experiment

    def add_trial(self, name, desc, config, experiment_id):
        """
        Add a new trial to the database.
        """
        session = self.Session()
        trial = Trial(
            id=generate_id(),
            name=name,
            desc=desc,
            start_time=datetime.now(),
            update_time=datetime.now(),
            config=json.dumps(config),
            experiment_id=experiment_id
        )
        session.add(trial)
        session.commit()
        session.close()
        return trial

    def add_trial_run(self, trial_id):
        """
        Add a new trial run to the database.
        """
        session = self.Session()
        trial_run = TrialRun(
            id=generate_id(),
            start_time=datetime.now(),
            status="STARTED",
            trial_id=trial_id
        )
        session.add(trial_run)
        session.commit()
        session.close()
        return trial_run

    def update_trial_run(self, trial_run_id, status, end_time=None):
        """
        Update the status and end time of a trial run.
        """
        session = self.Session()
        trial_run = session.query(TrialRun).filter_by(id=trial_run_id).first()
        if trial_run:
            trial_run.status = status
            if end_time:
                trial_run.end_time = end_time
            session.commit()
        session.close()

    def add_results(self, trial_run_id, total_accuracy, accuracy_per_label, total_loss, loss_per_label):
        """
        Add results for a trial run.
        """
        session = self.Session()
        results = Results(
            id=generate_id(),
            trial_run_id=trial_run_id,
            total_accuracy=total_accuracy,
            accuracy_per_label=json.dumps(accuracy_per_label),
            total_loss=total_loss,
            loss_per_label=json.dumps(loss_per_label)
        )
        session.add(results)
        session.commit()
        session.close()
        return results

    def add_epoch(self, trial_run_id, index, total_accuracy, accuracy_per_label, total_loss, loss_per_label):
        """
        Add an epoch for a trial run.
        """
        session = self.Session()
        epoch = Epoch(
            id=generate_id(),
            trial_run_id=trial_run_id,
            index=index,
            total_accuracy=total_accuracy,
            accuracy_per_label=json.dumps(accuracy_per_label),
            total_loss=total_loss,
            loss_per_label=json.dumps(loss_per_label)
        )
        session.add(epoch)
        session.commit()
        session.close()
        return epoch

    def add_dataset(self, size, location, config):
        """
        Add a new dataset to the database.
        """
        session = self.Session()
        dataset = Dataset(
            id=generate_id(),
            size=size,
            location=location,
            config=json.dumps(config)
        )
        session.add(dataset)
        session.commit()
        session.close()
        return dataset

    def add_encoder(self, type, config):
        """
        Add a new encoder to the database.
        """
        session = self.Session()
        encoder = Encoder(
            type=type,
            config=json.dumps(config)
        )
        session.add(encoder)
        session.commit()
        session.close()
        return encoder

    def add_artifact(self, type, config, location, trial_run_id=None, epoch_id=None, results_id=None):
        """
        Add a new artifact to the database.
        """
        if not os.path.exists(os.path.join(self.data_path, location)):
            raise FileNotFoundError(f"Artifact location '{location}' does not exist.")
        
        session = self.Session()
        artifact = Artifact(
            id=generate_id(),
            type=type,
            config=json.dumps(config),
            location=location,
            trial_run_id=trial_run_id,
            epoch_id=epoch_id,
            results_id=results_id
        )
        session.add(artifact)
        session.commit()
        session.close()
        return artifact

    def add_logs(self, location, trial_run_id):
        """
        Add logs for a trial run.
        """
        session = self.Session()
        logs = Logs(
            id=generate_id(),
            location=location,
            trial_run_id=trial_run_id
        )
        session.add(logs)
        session.commit()
        session.close()
        return logs

def generate_id():
    """
    Generate a unique ID for database entries.
    """
    import uuid
    return str(uuid.uuid4())

