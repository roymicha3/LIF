from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from experiment.db.tables import Base, Experiment, Trial, TrialRun, Results, Epoch, Dataset, Encoder, Artifact, Logs

class DBManager:
    """
    Database Manager to interact with the experiment tracking database.
    Provides methods to create, update, delete, and retrieve records from the database.
    """
    
    def __init__(self, db_uri='sqlite:///experiment_tracking.db'):
        """Initialize the database connection and session."""
        self.engine = create_engine(db_uri, echo=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def create_experiment(self, exp_id, title, desc, start_time, update_time, config, dataset_id):
        """Create a new experiment record."""
        session = self.Session()
        experiment = Experiment(id=exp_id, title=title, desc=desc, start_time=start_time,
                                update_time=update_time, config=config, dataset_id=dataset_id)
        session.add(experiment)
        session.commit()
        session.close()
    
    def get_experiment(self, exp_id):
        """Retrieve an experiment by ID."""
        session = self.Session()
        experiment = session.query(Experiment).filter_by(id=exp_id).first()
        session.close()
        return experiment
    
    def delete_experiment(self, exp_id):
        """Delete an experiment by ID."""
        session = self.Session()
        experiment = session.query(Experiment).filter_by(id=exp_id).first()
        if experiment:
            session.delete(experiment)
            session.commit()
        session.close()
    
    def create_trial(self, trial_id, name, desc, start_time, update_time, config, experiment_id):
        """Create a new trial record."""
        session = self.Session()
        trial = Trial(id=trial_id, name=name, desc=desc, start_time=start_time,
                      update_time=update_time, config=config, experiment_id=experiment_id)
        session.add(trial)
        session.commit()
        session.close()
    
    def get_trial(self, trial_id):
        """Retrieve a trial by ID."""
        session = self.Session()
        trial = session.query(Trial).filter_by(id=trial_id).first()
        session.close()
        return trial
    
    def create_trial_run(self, run_id, start_time, end_time, status, trial_id):
        """Create a new trial run record."""
        session = self.Session()
        trial_run = TrialRun(id=run_id, start_time=start_time, end_time=end_time, status=status, trial_id=trial_id)
        session.add(trial_run)
        session.commit()
        session.close()
    
    def get_trial_run(self, run_id):
        """Retrieve a trial run by ID."""
        session = self.Session()
        trial_run = session.query(TrialRun).filter_by(id=run_id).first()
        session.close()
        return trial_run
    
    def create_results(self, result_id, trial_run_id, total_accuracy, accuracy_per_label, total_loss, loss_per_label):
        """Create a new results record."""
        session = self.Session()
        results = Results(id=result_id, trial_run_id=trial_run_id, total_accuracy=total_accuracy,
                          accuracy_per_label=accuracy_per_label, total_loss=total_loss, loss_per_label=loss_per_label)
        session.add(results)
        session.commit()
        session.close()
    
    def get_results(self, result_id):
        """Retrieve results by ID."""
        session = self.Session()
        results = session.query(Results).filter_by(id=result_id).first()
        session.close()
        return results
    
    def create_dataset(self, dataset_id, size, location, config):
        """Create a new dataset record."""
        session = self.Session()
        dataset = Dataset(id=dataset_id, size=size, location=location, config=config)
        session.add(dataset)
        session.commit()
        session.close()
    
    def get_dataset(self, dataset_id):
        """Retrieve a dataset by ID."""
        session = self.Session()
        dataset = session.query(Dataset).filter_by(id=dataset_id).first()
        session.close()
        return dataset
    
    def create_log(self, log_id, location, trial_run_id):
        """Create a new log entry."""
        session = self.Session()
        log = Logs(id=log_id, location=location, trial_run_id=trial_run_id)
        session.add(log)
        session.commit()
        session.close()
    
    def get_log(self, log_id):
        """Retrieve log by ID."""
        session = self.Session()
        log = session.query(Logs).filter_by(id=log_id).first()
        session.close()
        return log
    