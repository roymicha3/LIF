from analysis.plotting import Plotter

def test_plotter():
    plotter = Plotter("D:\\results\\DB\\experiment.db")
    df = plotter.load_experiment_metrics("data len", "val_acc")
    
    print(df)
    
    df = plotter.load_experiment_avg_epochs("data len")
    
    print(df)
    

if __name__ == "__main__":
    test_plotter()
