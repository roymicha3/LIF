"""
This class is responsible for plotting using the DB
"""

from experiment.db.database import DB
from experiment.db.tables import Experiment

import pandas as pd

class Plotter:
    
    def __init__(self, db_path: str) -> None:
        DB.initialize(db_path)
        
    
    def load_experiment_metrics(self, experiment_name: str) -> pd.DataFrame:
        experiment_id = DB.instance().get_experiment_id(experiment_name)

        # Use the session context manager to keep session alive while accessing relationships
        with DB.instance().session_scope() as session:
            experiment = session.query(Experiment).get(experiment_id)

            data = []
            for trial in experiment.trials:
                trial_name = trial.name
                for trial_run in trial.trial_runs:
                    trial_run_name = trial_run.trial.name
                    for epoch in trial_run.epochs:
                        epoch_idx = epoch.idx
                        for metric in epoch.metrics:
                            data.append({
                                'trial': trial_name,
                                'trial_run': trial_run_name,
                                'epoch': epoch_idx,
                                'metric': metric.type,
                                'value': metric.total_val
                            })

            # Create DataFrame and handle potential duplicates by taking the mean
            df = pd.DataFrame(data)
            # Group by all columns except 'value' and aggregate
            df = df.groupby(['trial', 'trial_run', 'epoch', 'metric'])['value'].mean().reset_index()
            # Now set the index and unstack
            df = df.set_index(['trial', 'trial_run', 'epoch', 'metric'])
            df = df.unstack('metric')
            
            # Clean up the column names (remove the 'value' level if it exists)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)

            return df
