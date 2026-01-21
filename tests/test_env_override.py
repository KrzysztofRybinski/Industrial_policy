import textwrap

from industrial_policy.config import load_config


def test_env_overrides_sec_user_agent(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    data_dir = tmp_path / "data"
    outputs_dir = tmp_path / "outputs"
    config_path.write_text(
        textwrap.dedent(
            """
            sec:
              user_agent: "FROM_YAML"
            project:
              data_dir: "{data_dir}"
              outputs_dir: "{outputs_dir}"
            """
        ).format(data_dir=data_dir.as_posix(), outputs_dir=outputs_dir.as_posix()),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEC_USER_AGENT", "FROM_ENV")

    config = load_config(config_path)

    assert config["sec"]["user_agent"] == "FROM_ENV"
