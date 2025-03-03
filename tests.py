from analysis.plotting import Plotter

def test_plotter():
    plotter = Plotter("D:\\results\\DB\\experiment.db")
    df = plotter.load_experiment_metrics("data len", "val_acc")
    
    print(df)
    
    fig, ax = plotter.plot_epoch_progression("data len", "val_acc")
    

if __name__ == "__main__":
    test_plotter()
