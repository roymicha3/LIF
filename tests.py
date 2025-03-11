import matplotlib.pyplot as plt

from analysis.plotting import Plotter

EXPERIMENT_NAME = "polynomial lr scheduler large dataset"

def test_plotter():
    plotter = Plotter("D:\\results\\DB\\experiment.db")
    df = plotter.load_experiment_metrics(EXPERIMENT_NAME, "val_acc")
    
    print(df)
    
    fig, ax = plotter.plot_epoch_progression(EXPERIMENT_NAME, "val_loss", average_runs=True)
    
    plt.show()

if __name__ == "__main__":
    test_plotter()
