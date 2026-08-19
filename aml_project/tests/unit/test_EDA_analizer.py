from src.features.edaAnalyzer import EDAAnalyzer


def test_target_distribution_counts(sample_df, mocker):
    fake_plotter = mocker.Mock()
    analyzer = EDAAnalyzer(sample_df, "salary", ["age", "hours-per-week"],
                           ["workclass"], fake_plotter)
    counts, pct = analyzer.target_distribution()
    assert counts.sum() == len(sample_df)
    assert abs(pct.sum() - 100) < 1e-6
    fake_plotter.countplot.assert_called_once()


def test_detect_outliers_iqr_returns_expected_keys(sample_df, mocker):
    fake_plotter = mocker.Mock()
    num_cols = ["age", "hours-per-week"]
    analyzer = EDAAnalyzer(sample_df, "salary", num_cols, [
                           "workclass"], fake_plotter)
    report = analyzer.detect_outliers_iqr()
    assert set(report.keys()) == set(num_cols)
    for col_report in report.values():
        assert "count" in col_report and "pct" in col_report
