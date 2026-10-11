//! Helpers shared by the CLI integration tests.

// Each test crate compiles its own copy and uses only part of it.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

pub fn example(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(rel)
}

/// Fresh per-test dir; unique per process so concurrent runs don't clobber each other.
pub fn workdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("duello_{name}_{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn duello(dir: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_duello"));
    cmd.current_dir(dir);
    cmd
}

pub fn run(cmd: &mut Command) {
    let out = cmd.output().unwrap();
    assert!(
        out.status.success(),
        "duello failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
