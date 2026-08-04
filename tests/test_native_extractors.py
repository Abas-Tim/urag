"""Tests for native-language extractors (go, rust, java, c, cpp)."""

from urag.extractors.native_ext import GoExtractor, RustExtractor, JavaExtractor, CExtractor


GO_SRC = '''package auth

import "strings"

// ParseToken parses a raw token.
func ParseToken(raw string) (string, error) {
    return strings.TrimSpace(raw), nil
}

type TokenValidator struct {
    TTL int
}

func (v *TokenValidator) Validate(token string) bool {
    return token != ""
}

type Validatable interface {
    Validate(token string) bool
}
'''


def test_go():
    units = GoExtractor().extract(GO_SRC, "auth.go")
    types = [(u.unit_type, u.name, u.qualname) for u in units]
    assert ("function", "ParseToken", "ParseToken") in types
    assert ("struct", "TokenValidator", "TokenValidator") in types
    assert ("method", "Validate", "TokenValidator.Validate") in types
    assert ("interface", "Validatable", "Validatable") in types
    assert any(t[0] == "import" and "strings" in t[1] for t in types)
    pt = next(u for u in units if u.qualname == "ParseToken")
    assert "ParseToken" in pt.summary


RUST_SRC = '''use std::collections::HashMap;

/// Parses a raw token.
pub fn parse_token(raw: &str) -> Result<String, Error> {
    Ok(raw.trim().to_string())
}

pub struct TokenValidator {
    ttl: u64,
}

impl TokenValidator {
    pub fn new() -> Self {
        TokenValidator { ttl: 60 }
    }

    pub fn validate(&self, token: &str) -> bool {
        !token.is_empty()
    }
}

pub trait Validatable {
    fn validate(&self, token: &str) -> bool;
}

pub enum TokenKind {
    Bearer,
    ApiKey,
}
'''


def test_rust():
    units = RustExtractor().extract(RUST_SRC, "auth.rs")
    types = [(u.unit_type, u.name, u.qualname) for u in units]
    assert ("function", "parse_token", "parse_token") in types
    assert ("struct", "TokenValidator", "TokenValidator") in types
    assert ("method", "validate", "TokenValidator::validate") in types
    assert ("method", "new", "TokenValidator::new") in types
    assert ("trait", "Validatable", "Validatable") in types
    assert ("enum", "TokenKind", "TokenKind") in types
    pt = next(u for u in units if u.qualname == "parse_token")
    assert "Parses a raw token" in pt.summary


JAVA_SRC = '''package com.example.auth;

import java.util.List;

/**
 * Validates JWT tokens.
 */
public class TokenValidator {
    private int ttl = 60;

    public boolean validate(String token) {
        return token != null;
    }

    public static TokenValidator create() {
        return new TokenValidator();
    }
}

public interface Validatable {
    boolean validate(String token);
}

public enum TokenKind {
    BEARER, API_KEY
}
'''


def test_java():
    units = JavaExtractor().extract(JAVA_SRC, "Auth.java")
    types = [(u.unit_type, u.name, u.qualname) for u in units]
    assert ("class", "TokenValidator", "TokenValidator") in types
    assert ("method", "validate", "TokenValidator.validate") in types
    assert ("method", "create", "TokenValidator.create") in types
    assert ("interface", "Validatable", "Validatable") in types
    assert ("enum", "TokenKind", "TokenKind") in types
    tv = next(u for u in units if u.qualname == "TokenValidator")
    assert "JWT" in tv.summary


C_SRC = '''#include <stdio.h>

/* Validates a raw token. */
int validate_token(const char *token) {
    return token != NULL;
}

typedef struct {
    int ttl;
} TokenValidator;

struct TokenStore {
    char *name;
    int (*save)(const char *);
};
'''


def test_c():
    units = CExtractor("c").extract(C_SRC, "auth.c")
    types = [(u.unit_type, u.name, u.qualname) for u in units]
    assert ("function", "validate_token", "validate_token") in types
    assert ("typedef", "TokenValidator", "TokenValidator") in types
    assert ("struct", "TokenStore", "TokenStore") in types
    assert any(t[0] == "import" and "stdio" in t[1] for t in types)
    vt = next(u for u in units if u.qualname == "validate_token")
    assert "Validates a raw token" in vt.summary


CPP_SRC = '''#include <string>

namespace auth {

class TokenValidator {
public:
    bool validate(const std::string& token);
private:
    int ttl_ = 60;
};

template <typename T>
T parse_token(const std::string& raw) {
    return T();
}

}  // namespace auth
'''


def test_cpp():
    units = CExtractor("cpp").extract(CPP_SRC, "auth.cpp")
    types = [(u.unit_type, u.name, u.qualname) for u in units]
    assert ("class", "TokenValidator", "auth::TokenValidator") in types
    assert ("method", "validate", "auth::TokenValidator::validate") in types
    assert ("function", "parse_token", "auth::parse_token") in types
