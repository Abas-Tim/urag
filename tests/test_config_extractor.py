"""Tests for the configuration-file extractor (json, yaml, toml, ini, env)."""

from urag.extractors.config_ext import ConfigExtractor

JSON_SRC = """{
  "server": {
    "host": "0.0.0.0",
    "port": 8080
  },
  "database": {
    "url": "postgres://localhost/db"
  }
}
"""

YAML_SRC = """server:
  host: 0.0.0.0
  port: 8080
database:
  url: postgres://localhost/db
"""

TOML_SRC = """[server]
host = "0.0.0.0"
port = 8080

[database]
url = "postgres://localhost/db"
"""

INI_SRC = """[server]
host=0.0.0.0
port=8080

[database]
url=postgres://localhost/db
"""

ENV_SRC = """# app settings
export SERVER_HOST=0.0.0.0
SERVER_PORT=8080
DATABASE_URL=postgres://localhost/db
"""


def _qualnames(units):
    return {u.qualname for u in units}


def test_json_keys():
    units = ConfigExtractor("json").extract(JSON_SRC, "config.json")
    names = _qualnames(units)
    assert "server.host" in names
    assert "server.port" in names
    assert "database.url" in names
    host = next(u for u in units if u.qualname == "server.host")
    assert host.unit_type == "config_key"
    assert host.summary == '"0.0.0.0"'


def test_yaml_keys():
    units = ConfigExtractor("yaml").extract(YAML_SRC, "settings.yaml")
    names = _qualnames(units)
    assert "server.host" in names
    assert "server.port" in names
    assert "database.url" in names


def test_toml_keys():
    units = ConfigExtractor("toml").extract(TOML_SRC, "config.toml")
    names = _qualnames(units)
    assert "server.host" in names
    assert "server.port" in names
    assert "database.url" in names


def test_ini_keys():
    units = ConfigExtractor("ini").extract(INI_SRC, "app.ini")
    names = _qualnames(units)
    assert "server.host" in names
    assert "server.port" in names
    assert "database.url" in names


def test_env_keys():
    units = ConfigExtractor("env").extract(ENV_SRC, ".env")
    names = _qualnames(units)
    assert "SERVER_HOST" in names
    assert "SERVER_PORT" in names
    assert "DATABASE_URL" in names
    # export-prefixed and bare lines both parse
    host = next(u for u in units if u.qualname == "SERVER_HOST")
    assert host.summary == "0.0.0.0"


def test_config_units_have_spans():
    units = ConfigExtractor("json").extract(JSON_SRC, "config.json")
    for u in units:
        assert u.start_line == u.end_line
        assert u.byte_start <= u.byte_end
