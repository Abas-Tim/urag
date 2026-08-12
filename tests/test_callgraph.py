"""Tests for call-graph extraction and impact routing."""

from urag.extractors.python_ext import PythonExtractor
from urag.extractors.ts_ext import TsExtractor
from urag.extractors.native_ext import (
    GoExtractor,
    JavaExtractor,
    CSharpExtractor,
    CExtractor,
)
from urag.retrieve import Retriever
from urag.classify import classify


def test_python_calls():
    src = """import os

class Auth:
    def validate(self, token):
        return os.path.exists(token)

    def run(self):
        return self.validate("x")

def main():
    a = Auth()
    a.run()
"""
    calls = PythonExtractor().collect_calls(src)
    names = {(c.callee, c.callee_full, c.line) for c in calls}
    assert ("validate", "self.validate", 8) in names
    assert ("exists", "os.path.exists", 5) in names
    assert ("run", "a.run", 12) in names
    assert ("Auth", "Auth", 11) in names  # constructor call


def test_go_calls():
    src = """package main

func main() {
    handleRequest("GET", "/x")
}

func handleRequest(method, path string) {
    parseBody(path)
}
"""
    calls = GoExtractor().collect_calls(src)
    names = {(c.callee, c.callee_full, c.line) for c in calls}
    assert ("handleRequest", "handleRequest", 4) in names
    assert ("parseBody", "parseBody", 8) in names


def test_java_calls():
    src = """public class A {
    void run() {
        B helper = new B();
        helper.process("x");
        process("y");
    }
    void process(String s) {}
}
"""
    calls = JavaExtractor().collect_calls(src)
    names = {(c.callee, c.callee_full) for c in calls}
    assert ("process", "helper.process") in names
    assert ("process", "process") in names


def test_csharp_calls():
    src = """public class A {
    void Run() {
        var b = new B();
        b.Process("x");
        Finish();
    }
    void Finish() {}
}
"""
    calls = CSharpExtractor().collect_calls(src)
    names = {(c.callee, c.callee_full) for c in calls}
    assert ("Process", "b.Process") in names
    assert ("Finish", "Finish") in names


def test_cextractor():
    src = """int run() {
    return helper(1) + finish();
}
int helper(int x) { return x; }
int finish() { return 0; }
"""
    calls = CExtractor("c").collect_calls(src)
    names = {(c.callee, c.callee_full) for c in calls}
    assert ("helper", "helper") in names
    assert ("finish", "finish") in names


def test_csharp_units():
    src = """namespace App {
public class Service {
    public void Start() {
        Log("up");
    }
    private void Log(string m) {}
}
}
"""
    units = CSharpExtractor().extract(src, "svc.cs")
    types = [(u.unit_type, u.name, u.qualname) for u in units]
    assert ("class", "Service", "App.Service") in types
    assert ("method", "Start", "App.Service.Start") in types
    assert ("method", "Log", "App.Service.Log") in types


def test_classify_impact():
    assert classify("what calls TokenValidator") == "impact"
    assert classify("who uses validate") == "impact"
    assert classify("what breaks if I change parse_token") == "impact"


def test_impact_symbol():
    assert Retriever._impact_symbol("what calls TokenValidator") == "TokenValidator"
    assert Retriever._impact_symbol("who uses validate") == "validate"
    assert (
        Retriever._impact_symbol("what breaks if I change parse_token") == "parse_token"
    )
