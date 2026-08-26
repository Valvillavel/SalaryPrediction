from src.visualization.plots import PlotHelper


def test_countplot_creates_file(sample_df, tmp_path):
    plotter = PlotHelper(tmp_path)
    plotter.countplot(sample_df, "salary", "test_countplot.png")
    assert (tmp_path / "test_countplot.png").exists()


def test_confusion_matrix_creates_file(tmp_path):
    import numpy as np
    plotter = PlotHelper(tmp_path)
    cm = np.array([[5, 1], [2, 4]])
    plotter.confusion_matrix(cm, [" <=50K", " >50K"], "Modelo Dummy", "cm.png")
    assert (tmp_path / "cm.png").exists()
