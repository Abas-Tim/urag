from pathlib import Path

from urag.config import language_for_path
from urag.extractors.markdown_ext import MarkdownExtractor
from urag.extractors.python_ext import PythonExtractor


def test_unicode_extractor_preserves_names():
    units = PythonExtractor().extract("def café(token):\n    return token\n", "x.py")
    assert units[0].name == "café"
    assert "café" in units[0].signature


def test_markdown_preserves_preamble_and_ignores_fenced_headings():
    source = "Preamble\n\n```md\n# not a heading\n```\n# Real\nBody\n"
    units = MarkdownExtractor().extract(source, "README.md")
    assert units[0].qualname == "#README.md#preamble"
    assert units[0].start_line == 1
    assert [unit.name for unit in units] == ["", "# Real"]


def test_tsx_is_a_distinct_extractor_language():
    assert language_for_path(Path("component.tsx")) == ("tsx", "source")
