"""Smoke tests for extractors."""

from urag.extractors.python_ext import PythonExtractor
from urag.extractors.ts_ext import TsExtractor
from urag.extractors.markdown_ext import MarkdownExtractor
from urag.extractors.base import ByteIndexedSource


def test_byte_indexed_source_uses_utf8_byte_ranges():
    source = ByteIndexedSource("éclair")

    assert source[0:2] == "é"


def test_python_extractor():
    src = '''"""Module doc."""
import os
from collections import defaultdict

def parse_config(path: str, defaults: dict | None = None) -> dict:
    """Parse a config file into a dict."""
    return {}

class TokenValidator:
    """Validates JWT tokens."""

    def validate(self, token: str) -> bool:
        """Check expiration."""
        return True

    def _helper(self):
        pass
'''
    units = PythonExtractor().extract(src, "auth.py")
    types = [(u.unit_type, u.name, u.qualname) for u in units]
    assert ("function", "parse_config", "parse_config") in types
    assert ("class", "TokenValidator", "TokenValidator") in types
    assert ("method", "validate", "TokenValidator.validate") in types
    assert ("import", "defaultdict", "collections.defaultdict") in types
    v = next(u for u in units if u.qualname == "TokenValidator.validate")
    assert "expiration" in v.summary
    assert v.parent_id is None


def test_python_extractor_tolerates_malformed_definitions():
    src = "def broken(:\n    pass\nclass Broken(\n"

    units = PythonExtractor().extract(src, "broken.py")

    assert [(unit.unit_type, unit.name) for unit in units] == [("function", "broken")]
    assert units[0].byte_start == 0
    assert units[0].byte_end == len("def broken(:\n    pass".encode("utf-8"))


def test_ts_extractor():
    src = """import { readFile } from "fs";

/**
 * Validates JWT tokens.
 */
export interface TokenValidator {
  validate(token: string): boolean;
}

export class Validator implements TokenValidator {
  validate(token: string): boolean {
    return true;
  }
}

type Callback = (err: Error | null) => void;
const helper = (x: number) => x * 2;
export const factory = () => new Validator();
"""
    units = TsExtractor("typescript").extract(src, "auth.ts")
    types = [(u.unit_type, u.name, u.qualname) for u in units]
    assert ("interface", "TokenValidator", "TokenValidator") in types
    assert ("method", "validate", "TokenValidator.validate") in types
    assert ("class", "Validator", "Validator") in types
    assert ("method", "validate", "Validator.validate") in types
    assert ("function", "factory", "factory") in types
    assert any(t[1] == "readFile" for t in types)
    v = next(u for u in units if u.qualname == "TokenValidator")
    assert "JWT" in v.summary


def test_markdown_extractor():
    src = """# Project

Intro paragraph about the project.

## Design

Token validation uses a signing key.

### Details

Deep dive here.
"""
    units = MarkdownExtractor().extract(src, "README.md")
    assert len(units) == 3
    assert units[0].name == "# Project"
    assert "Intro paragraph" in units[0].summary
    assert units[1].name == "## Design"
    assert "signing key" in units[1].summary
    assert units[1].qualname == "#README.md#Design"


def test_markdown_offsets_include_multibyte_lines():
    src = "# Café\n\nRésumé\n"

    unit = MarkdownExtractor().extract(src, "README.md")[0]

    assert unit.byte_start == 0
    assert unit.byte_end == len(src.encode("utf-8"))
