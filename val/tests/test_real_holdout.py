from val.real_holdout import CLEAN_CATEGORIES, strip_dup_suffix, trained_base_ids


def test_strip_dup_suffix():
    assert strip_dup_suffix("00066667") == "00066667"
    assert strip_dup_suffix("00066667-p0") == "00066667"
    assert strip_dup_suffix("abc123-d11") == "abc123"


def test_trained_base_ids_reads_index(tmp_path):
    index = tmp_path / "index.jsonl"
    index.write_text(
        '{"problem_id": "aaa", "category": "numeral"}\n'
        '{"problem_id": "aaa-d0", "category": "numeral"}\n'
        '{"problem_id": "bbb-p0", "category": "cipher"}\n'
    )
    assert trained_base_ids(index) == {"aaa", "bbb"}


def test_clean_categories_are_the_downsampled_three():
    assert CLEAN_CATEGORIES == {"numeral", "gravity", "unit_conversion"}
