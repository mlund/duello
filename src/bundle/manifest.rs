//! `manifest.json`: what the archive holds, and the format version readers check.

use super::zipio::{Entry, Reader};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use zip::DateTime;

const PATH: &str = "manifest.json";
const FORMAT: &str = "duello-session";
/// Bumped on any layout or JSON shape change; readers refuse other versions.
pub const FORMAT_VERSION: u32 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Part {
    Twobody,
    Manybody,
    Trajectory,
}

#[derive(Serialize, Deserialize)]
pub struct Manifest {
    format: String,
    version: u32,
    /// Writer's duello version.
    duello: String,
    /// UTC, ISO 8601.
    created: String,
    pub parts: Vec<Part>,
}

impl Manifest {
    pub fn new(parts: Vec<Part>, created: &Civil) -> Self {
        Self {
            format: FORMAT.into(),
            version: FORMAT_VERSION,
            duello: crate::VERSION.into(),
            created: created.iso(),
            parts,
        }
    }

    pub fn created(&self) -> &str {
        &self.created
    }

    pub fn entry(&self) -> Result<Entry<'static>> {
        Entry::json(PATH, self, "Format, version and the parts present")
    }

    /// Checks the version before the rest, whose shape may differ in others.
    pub fn read(zip: &mut Reader) -> Result<Self> {
        let json: serde_json::Value = zip.json(PATH)?;
        let head = Head::deserialize(&json).context(PATH)?;
        ensure!(head.format == FORMAT, "not a duello session file");
        match head.version.cmp(&FORMAT_VERSION) {
            Ordering::Less => bail!(
                "session format version {} is older than this duello reads ({FORMAT_VERSION}); re-run the scan",
                head.version
            ),
            Ordering::Greater => bail!(
                "session format version {} is newer than this duello reads ({FORMAT_VERSION}); update duello",
                head.version
            ),
            Ordering::Equal => Ok(Self::deserialize(json).context(PATH)?),
        }
    }
}

/// The part every version shares.
#[derive(Deserialize)]
struct Head {
    format: String,
    version: u32,
}

/// The `.duello` container before sessions were zips.
const OLD_MAGIC: &[u8] = b"DUELLO-SCAN\0";

pub fn refuse_old_format(bytes: &[u8]) -> Result<()> {
    ensure!(
        !bytes.starts_with(OLD_MAGIC),
        "older duello format; re-run the scan"
    );
    Ok(())
}

/// A UTC calendar time; zip and ISO 8601 both want one, and std has none.
pub struct Civil([u32; 6]);

pub fn now() -> Civil {
    let seconds = web_time::SystemTime::now()
        .duration_since(web_time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    Civil::from_unix(seconds)
}

impl Civil {
    /// Howard Hinnant's days-to-civil algorithm.
    fn from_unix(seconds: u64) -> Self {
        let (days, rest) = ((seconds / 86_400) as i64, (seconds % 86_400) as u32);
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
        let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
        let year = (yoe + era * 400 + i64::from(month <= 2)) as u32;
        Self([year, month, day, rest / 3600, rest / 60 % 60, rest % 60])
    }

    fn iso(&self) -> String {
        let [y, mo, d, h, mi, s] = self.0;
        format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
    }

    /// Falls back to the zip epoch outside zip's 1980–2107 range.
    pub fn zip(&self) -> DateTime {
        let [year, rest @ ..] = self.0;
        let [mo, d, h, mi, s] = rest.map(|x| x as u8);
        let year = u16::try_from(year).unwrap_or(0);
        DateTime::from_date_and_time(year, mo, d, h, mi, s).unwrap_or_default()
    }
}
