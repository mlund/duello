//! Zip entries in and out: compression per entry, size limits, and errors
//! that name the entry.

use anyhow::{anyhow, Context, Result};
use serde::{de::DeserializeOwned, Serialize};
use std::borrow::Cow;
use std::io::{Cursor, Read, Write};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, DateTime, ZipArchive, ZipWriter};

/// How an entry is compressed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Packing {
    Deflated,
    /// Large and written once; the fastest level keeps saving responsive.
    DeflatedFast,
    /// Already compressed, so deflating again only costs time.
    Stored,
}

/// One file of the archive, with what a reader needs to know about it.
/// Borrows large inputs such as the trajectory, which would otherwise be
/// held twice while zipping.
pub struct Entry<'a> {
    pub path: String,
    pub bytes: Cow<'a, [u8]>,
    pub packing: Packing,
    /// MIME type, for the RO-Crate.
    pub format: &'static str,
    pub description: String,
}

impl<'a> Entry<'a> {
    pub fn json(path: &str, value: &impl Serialize, description: &str) -> Result<Self> {
        let bytes = serde_json::to_vec_pretty(value).with_context(|| path.to_string())?;
        Ok(Self::new(path, bytes, "application/json", description))
    }

    pub fn text(path: &str, text: String, format: &'static str, description: &str) -> Self {
        Self::new(path, text.into_bytes(), format, description)
    }

    pub fn new(
        path: &str,
        bytes: impl Into<Cow<'a, [u8]>>,
        format: &'static str,
        description: &str,
    ) -> Self {
        Self {
            path: path.to_string(),
            bytes: bytes.into(),
            packing: Packing::Deflated,
            format,
            description: description.to_string(),
        }
    }

    pub fn packed(self, packing: Packing) -> Self {
        Self { packing, ..self }
    }
}

pub fn write_zip(entries: &[Entry], modified: DateTime) -> Result<Vec<u8>> {
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    for entry in entries {
        let options = options(entry.packing, modified)
            // Zip64 only where needed, as some unzip tools still lack it.
            .large_file(entry.bytes.len() >= u32::MAX as usize);
        zip.start_file(entry.path.as_str(), options)
            .with_context(|| entry.path.clone())?;
        zip.write_all(&entry.bytes)?;
    }
    Ok(zip.finish()?.into_inner())
}

fn options(packing: Packing, modified: DateTime) -> SimpleFileOptions {
    let options = SimpleFileOptions::default().last_modified_time(modified);
    match packing {
        Packing::Deflated => options.compression_method(CompressionMethod::Deflated),
        Packing::DeflatedFast => options
            .compression_method(CompressionMethod::Deflated)
            .compression_level(Some(1)),
        Packing::Stored => options.compression_method(CompressionMethod::Stored),
    }
}

/// Bounds what one entry may inflate to, so a crafted file can't exhaust
/// memory: well above a 256 MiB trajectory or a fine 6D table.
const MAX_ENTRY_BYTES: u64 = 4 << 30;

/// Most a header may make us allocate before the bytes back it up.
const RESERVE_LIMIT: u64 = 256 << 20;

pub struct Reader<'a> {
    zip: ZipArchive<Cursor<&'a [u8]>>,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let zip = ZipArchive::new(Cursor::new(bytes)).context("not a duello session file")?;
        Ok(Self { zip })
    }

    pub fn bytes(&mut self, path: &str) -> Result<Vec<u8>> {
        let file = self
            .zip
            .by_name(path)
            .map_err(|_| anyhow!("session file lacks {path}"))?;
        // The header's size is a crafted file's to choose, so only trust it so far.
        let mut bytes = Vec::with_capacity(file.size().min(RESERVE_LIMIT) as usize);
        file.take(MAX_ENTRY_BYTES + 1)
            .read_to_end(&mut bytes)
            .with_context(|| path.to_string())?;
        anyhow::ensure!(bytes.len() as u64 <= MAX_ENTRY_BYTES, "{path} is too large");
        Ok(bytes)
    }

    pub fn text(&mut self, path: &str) -> Result<String> {
        String::from_utf8(self.bytes(path)?).with_context(|| format!("{path} is not UTF-8"))
    }

    pub fn json<T: DeserializeOwned>(&mut self, path: &str) -> Result<T> {
        serde_json::from_slice(&self.bytes(path)?).with_context(|| path.to_string())
    }
}
