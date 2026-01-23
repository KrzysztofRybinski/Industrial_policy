from industrial_policy.utils.sec import normalize_cik


def test_normalize_cik_handles_numeric_and_strings() -> None:
    assert normalize_cik(320193) == "0000320193"
    assert normalize_cik("0000320193") == "0000320193"
    assert normalize_cik(" 1234 ") == "0000001234"
    assert normalize_cik("1234.0") == "0000001234"
    assert normalize_cik("CIK00001234") == "0000001234"
    assert normalize_cik("") is None
    assert normalize_cik(None) is None
