from industrial_policy.utils.textnorm import normalize_name


def test_normalize_name_removes_suffixes() -> None:
    assert normalize_name("Acme, Inc.") == "ACME"
    assert normalize_name("Beta Corporation") == "BETA"


def test_normalize_name_uppercase_and_punctuation() -> None:
    assert normalize_name("Foo-Bar LLC") == "FOO BAR"
