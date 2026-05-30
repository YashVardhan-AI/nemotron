from val.run_local_demo import run_demo


def test_demo_runs_and_scores(tmp_path):
    report = run_demo(
        n=5,
        difficulty=4,
        holdout_path=tmp_path / "holdout_rules.json",
        report_path=tmp_path / "demo_report.json",
    )
    assert report["cipher"]["overall"]["n"] == 5
    # Oracle returns ground-truth answers -> full accuracy.
    assert report["cipher"]["overall"]["accuracy"] == 1.0
    assert (tmp_path / "demo_report.json").exists()
    assert (tmp_path / "holdout_rules.json").exists()
