import matplotlib.pyplot as plt

from analysis.plotting import Plotter

EXPERIMENT_NAME = "data len 2"

def plot_metric_progression(database_path: str, experiment_name: str, metric: str, average_runs: bool = False):
    plotter = Plotter(database_path)
    
    df = plotter.load_experiment_metrics(experiment_name, metric)
    fig, ax = plotter.plot_epoch_progression(experiment_name, metric, average_runs=average_runs)
    
    plt.show()
    


if __name__ == "__main__":
    # plot_metric_progression(database_path="D:\\results\\DB\\experiment.db", experiment_name=EXPERIMENT_NAME, metric="val_loss", average_runs=True)

