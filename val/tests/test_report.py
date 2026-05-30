import json

from val.report import aggregate, format_table, write_json
from val.scoring import Result


def _r(category, difficulty, correct, correct_strict):
    return Result(
        id="x",
        category=category,
        difficulty=difficulty,
        answer="a",
        predicted="a",
        correct=correct,
        correct_strict=correct_strict,
        raw_text="",
    )


def test_aggregate_groups_by_category_and_difficulty():
    results = [
        _r("cipher", 1, True, True),
        _r("cipher", 1, False, False),
        _r("cipher", 2, True, False),
    ]
    agg = aggregate(results)

    assert agg["cipher"]["overall"]["n"] == 3
    assert agg["cipher"]["overall"]["accuracy"] == 2 / 3
    assert agg["cipher"]["overall"]["accuracy_strict"] == 1 / 3
    assert agg["cipher"]["by_difficulty"][1]["n"] == 2
    assert agg["cipher"]["by_difficulty"][1]["accuracy"] == 0.5
    assert agg["cipher"]["by_difficulty"][2]["accuracy"] == 1.0


def test_format_table_mentions_category_and_counts():
    table = format_table(aggregate([_r("cipher", 1, True, True)]))
    assert "cipher" in table
    assert "100" in table  # 100.0% accuracy somewhere in the row


def test_write_json_roundtrips(tmp_path):
    agg = aggregate([_r("cipher", 1, True, True)])
    out = tmp_path / "report.json"
    write_json(agg, out)
    assert json.loads(out.read_text())["cipher"]["overall"]["n"] == 1
