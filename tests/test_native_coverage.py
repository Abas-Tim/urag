"""Regression tests for C/C++ coverage gaps: extern "C" blocks, header
API surface (typedef enums, prototypes), include-based dependents, C++ type
references, and the new GLSL/CMake/PowerShell language support."""

import pytest

from urag.config import language_for_path, load_config
from urag.db import Database
from urag.embed import NoopEmbedder
from urag.extractors.c_ext import CExtractor
from urag.extractors.cmake_ext import CMakeExtractor
from urag.extractors.powershell_ext import PowerShellExtractor
from urag.indexer import Indexer
from urag.retrieve import Retriever

HEADER_H = """#ifndef GJXL_GJXL_H_
#define GJXL_GJXL_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum gjxl_status {
  GJXL_OK = 0,
  GJXL_ERROR_MEMORY_BUDGET = 9,
} gjxl_status;

typedef struct gjxl_buffer {
  uint8_t* data;
  size_t size;
} gjxl_buffer;

gjxl_status gjxl_encode_rgb8(const uint8_t* samples, size_t size_bytes);

#ifdef __cplusplus
}
#endif

#endif
"""

CPP_EXTERN_C = """namespace gjxl {
int Helper() { return 1; }
}  // namespace gjxl

extern "C" {

int gjxl_encode_rgb8(const unsigned char* samples) {
  return gjxl::Helper();
}

int gjxl_encode_gray8(const unsigned char* samples) {
  return 0;
}

}  // extern "C"
"""


def test_c_header_api_surface():
    units = CExtractor("c").extract(HEADER_H, "include/gjxl/gjxl.h")
    types = {(u.unit_type, u.name) for u in units}
    assert ("typedef", "gjxl_status") in types
    assert ("typedef", "gjxl_buffer") in types
    assert ("function", "gjxl_encode_rgb8") in types
    status = next(u for u in units if u.name == "gjxl_status")
    assert "GJXL_ERROR_MEMORY_BUDGET" in status.signature


def test_cpp_extern_c_block():
    units = CExtractor("cpp").extract(CPP_EXTERN_C, "src/gjxl.cc")
    names = {u.name for u in units}
    assert "gjxl_encode_rgb8" in names
    assert "gjxl_encode_gray8" in names
    assert "Helper" in names


def test_cpp_type_references():
    refs = CExtractor("cpp").collect_references(
        "enum class TreeMode { kSingle };\n"
        "void pick(TreeMode mode) {\n"
        "  if (mode == TreeMode::kSingle) { }\n"
        "}\n"
    )
    tree_refs = [r for r in refs if r.target == "TreeMode"]
    assert tree_refs, "TreeMode usages must produce reference edges"


CXX_FILES = {
    "include/gjxl/gjxl.h": "typedef int gjxl_status;\nint gjxl_encode(const char* s);\n",
    "src/encoder/modular_l0.h": "struct TreeMode {};\n",
    "src/encoder/modular_l0.cc": '#include "encoder/modular_l0.h"\nvoid build() {}\n',
    "src/gjxl.cc": (
        '#include "gjxl/gjxl.h"\n#include "encoder/modular_l0.h"\n'
        "int gjxl_encode(const char* s) { return 0; }\n"
    ),
    "src/gpu/kernels/table_build.comp": "void main() { }\n",
}

CMAKE = """cmake_minimum_required(VERSION 3.24)
project(gjxl)

option(GJXL_BUILD_TESTS "Build tests" ON)
option(GJXL_ENABLE_GPU "Build the Vulkan backend and compute kernels" ON)

add_library(gjxl src/gjxl.cc)
add_executable(gjxl_tools tools/main.cc)
"""

PS1 = """# Runs the qualification pass.
function Invoke-Qualify($Path) {
    Write-Host $Path
}
"""


@pytest.fixture()
def cxx_project(tmp_path):
    for rel, text in CXX_FILES.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    (tmp_path / "CMakeLists.txt").write_text(CMAKE, encoding="utf-8")
    (tmp_path / "qualify.ps1").write_text(PS1, encoding="utf-8")
    cfg = load_config(tmp_path)
    cfg.embedding.provider = "none"
    cfg.save()
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, NoopEmbedder()).index_all()
    yield cfg, db
    db.close()


def test_language_mapping():
    from pathlib import Path

    assert language_for_path(Path("a/CMakeLists.txt")) == ("cmake", "config")
    assert language_for_path(Path("k/table.comp")) == ("glsl", "source")
    assert language_for_path(Path("s/run.ps1")) == ("powershell", "source")
    assert language_for_path(Path("i/param_layout.inc")) == ("cpp", "source")


def test_cmake_options_indexed(cxx_project):
    cfg, db = cxx_project
    rows = db.conn.execute(
        "SELECT qualname, summary FROM units WHERE unit_type='config_key' ORDER BY qualname"
    ).fetchall()
    names = {r["qualname"] for r in rows}
    assert {"GJXL_BUILD_TESTS", "GJXL_ENABLE_GPU"} <= names
    gpu = next(r for r in rows if r["qualname"] == "GJXL_ENABLE_GPU")
    assert "Vulkan" in gpu["summary"]


def test_glsl_and_powershell_indexed(cxx_project):
    cfg, db = cxx_project
    langs = {r["language"] for r in db.file_list()}
    assert "glsl" in langs and "powershell" in langs
    rows = db.conn.execute(
        "SELECT u.name, u.unit_type FROM units u JOIN files f ON f.id=u.file_id "
        "WHERE f.language='powershell'"
    ).fetchall()
    assert any(r["name"] == "Invoke-Qualify" for r in rows)


def test_dependents_include_suffix_match(cxx_project):
    cfg, db = cxx_project
    retriever = Retriever(cfg, db, NoopEmbedder())
    result = retriever.dependents("src/encoder/modular_l0.h")
    paths = {r["path"] for r in result["results"]}
    assert "src/encoder/modular_l0.cc" in paths
    assert "src/gjxl.cc" in paths


def test_dependents_symbol_fallback(cxx_project):
    cfg, db = cxx_project
    retriever = Retriever(cfg, db, NoopEmbedder())
    result = retriever.dependents("gjxl_encode")
    assert result["results"], "symbol dependents must resolve via def file"


def test_header_prototypes_resolvable(cxx_project):
    cfg, db = cxx_project
    retriever = Retriever(cfg, db, NoopEmbedder())
    result = retriever.resolve("gjxl_encode")
    assert result.results


def test_cmake_extractor_multiline():
    units = CMakeExtractor().extract(
        'option(GJXL_X\n  "long\n   help"\n  ON)\nset(FOO "bar")\n', "CMakeLists.txt"
    )
    names = {u.name for u in units}
    assert {"GJXL_X", "FOO"} <= names


def test_powershell_extractor_brace_match():
    units = PowerShellExtractor().extract(
        "# Alpha worker.\nfunction A {\n  if ($x) {\n    foo\n  }\n}\nfunction B { bar }\n",
        "s.ps1",
    )
    by_name = {u.name: u for u in units}
    assert set(by_name) == {"A", "B"}
    assert by_name["A"].start_line == 2
    assert by_name["A"].end_line == 6
    assert by_name["A"].summary == "Alpha worker."
