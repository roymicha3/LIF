"""
This class is responsible for plotting using the DB
"""
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from experiment.db.database import DB


class Plotter:
    
    def __init__(self, db_path: str) -> None:
        DB.initialize(db_path)
        
    
    def load_experiment_metrics(self, experiment_name: str, metric: str) -> pd.DataFrame:
        """
        Load the metrics of the experiment
        """
        
        sql_query = \
            f"""
            select ex.title, t.name as trial_name,tr.id as run_id, ep.idx, m.total_val
            From experiment ex 
            join trial t on ex.id = t.experiment_id 
            join trial_run tr on t.id = tr.trial_id 
            join epoch ep on tr.id = ep.trial_run_id
            join epoch_metric em on em.epoch_idx=ep.idx and em.epoch_trial_run_id = tr.id
            join metric m on m.id= em.metric_id
            where ex.title = :experiment_name
            and m.type = :metric
            """
        params = {'experiment_name': experiment_name, 'metric': metric}
        
        engine = DB.instance().engine
        with engine.connect() as connection:
            df = pd.read_sql(text(sql_query), connection, params=params, index_col=None)
        
        return df
    
    def load_experiment_avg_epochs(self, experiment_name: str) -> pd.DataFrame:
        """
        Load the average epochs of the experiment
        """
        
        sql_query = \
            f"""
            select t.name as trial_name, AVG(ep.idx) as avg_epochs, MAX(m.total_val) as max_metric
            From experiment ex 
            join trial t on ex.id = t.experiment_id 
            join trial_run tr on t.id = tr.trial_id 
            join epoch ep on tr.id = ep.trial_run_id
            join epoch_metric em on em.epoch_idx=ep.idx and em.epoch_trial_run_id = tr.id
            join metric m on m.id= em.metric_id
            where ex.title = :experiment_name
            GROUP BY t.name
            """
        params = {'experiment_name': experiment_name}
        
        engine = DB.instance().engine
        with engine.connect() as connection:
            df = pd.read_sql(text(sql_query), connection, params=params, index_col=None)
        
        return df
    
