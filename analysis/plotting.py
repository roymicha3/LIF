"""
This class is responsible for plotting using the DB
"""
import pandas as pd
import seaborn as sns
from sqlalchemy import text
import matplotlib.pyplot as plt

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
    
    def plot_trial_num_epochs(self,
                              experiment_name: str,
                              figsize=(12, 6), 
                              sort_by='avg_epochs',
                              ascending=True, 
                              color_metric=None,
                              cmap='viridis'):
        """
        Plot the average number of epochs per trial for a given experiment.
        
        Parameters:
        -----------
        experiment_name : str
            Name of the experiment to analyze
        figsize : tuple, optional
            Figure size (width, height)
        sort_by : str, optional
            Column to sort trials by ('avg_epochs' or 'max_metric')
        ascending : bool, optional
            Sort in ascending order if True, descending if False
        color_metric : str, optional
            Metric name to use for color coding bars (None for default coloring)
        cmap : str, optional
            Colormap to use when color_metric is provided
            
        Returns:
        --------
        fig, ax : matplotlib figure and axes objects
        """
        # Load the average epochs data
        df = self.load_experiment_avg_epochs(experiment_name)
        
        # Sort the dataframe
        df = df.sort_values(by=sort_by, ascending=ascending)
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine colors if color_metric is provided
        if color_metric:
            metric_df = self.load_experiment_metrics(experiment_name, color_metric)
            # Get max value per trial
            color_values = metric_df.groupby('trial_name')['total_val'].max()
            # Align with sorted df
            color_values = color_values.loc[df['trial_name']].values
            # Normalize color values
            norm = plt.Normalize(color_values.min(), color_values.max())
            colors = plt.cm.get_cmap(cmap)(norm(color_values))
            
            # Create the bar plot with color mapping
            bars = ax.bar(df['trial_name'], df['avg_epochs'], color=colors)
            
            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(f'Max {color_metric}')
        else:
            # Create a simple bar plot
            bars = ax.bar(df['trial_name'], df['avg_epochs'])
        
        # Add labels and title
        ax.set_xlabel('Trial Name')
        ax.set_ylabel('Average Number of Epochs')
        ax.set_title(f'Average Epochs per Trial for Experiment: {experiment_name}')
        
        # Add value annotations on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # Rotate x-axis labels if there are many trials
        if len(df) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        return fig, ax
    
    def plot_epoch_progression(self, experiment_name: str, metric: str, figsize=(14, 8)):
        """
        Plot the progression of a metric across epochs for each trial.
        
        Parameters:
        -----------
        experiment_name : str
            Name of the experiment to analyze
        metric : str
            Metric to plot (e.g., 'accuracy', 'loss')
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns:
        --------
        fig, ax : matplotlib figure and axes objects
        """
        # Load metrics data
        df = self.load_experiment_metrics(experiment_name, metric)
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each trial as a line
        sns.lineplot(data=df, x='idx', y='total_val', hue='trial_name', 
                 marker='o', markersize=4, ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{metric}')
        ax.set_title(f'{metric} Progression Across Epochs for Experiment: {experiment_name}')
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Improve legend
        ax.legend(title='Trial', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        return fig, ax
    
