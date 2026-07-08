// Package testenv loads live-test credentials from the repository's .env file
// (see .env.example). Offline tests never touch it; live tests skip when a
// key is absent.
package testenv

import (
	"bufio"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
)

var once sync.Once

func load() {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return
	}
	root := filepath.Join(filepath.Dir(file), "..", "..")
	f, err := os.Open(filepath.Join(root, ".env"))
	if err != nil {
		return
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.Trim(strings.TrimSpace(value), `"`)
		if value != "" && os.Getenv(key) == "" {
			os.Setenv(key, value)
		}
	}
	_ = scanner.Err()
}

// Key returns the named variable or skips the test when it is unset.
func Key(t *testing.T, name string) string {
	t.Helper()
	once.Do(load)
	v := os.Getenv(name)
	if v == "" {
		t.Skipf("%s not set; fill .env to run live tests", name)
	}
	return v
}

// Get returns the named variable, falling back to a default.
func Get(name, fallback string) string {
	once.Do(load)
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}
