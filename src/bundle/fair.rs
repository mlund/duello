//! README and RO-Crate metadata, both generated from the entry list so they
//! can't disagree with what the zip holds.

use super::zipio::Entry;
use anyhow::Result;
use serde_json::{json, Value};

/// What the README and RO-Crate say about one entry.
struct Listing<'a> {
    path: &'a str,
    format: &'a str,
    description: &'a str,
}

const README: Listing = Listing {
    path: "README.md",
    format: "text/markdown",
    description: "What each file is and how to reopen the session",
};
const RO_CRATE: Listing = Listing {
    path: "ro-crate-metadata.json",
    format: "application/ld+json",
    description: "RO-Crate 1.1 metadata",
};

/// `entries` with the README and RO-Crate put first; both list every entry,
/// themselves included.
pub fn with_metadata<'a>(entries: Vec<Entry<'a>>, created: &str) -> Result<Vec<Entry<'a>>> {
    let listings: Vec<Listing> = [README, RO_CRATE]
        .into_iter()
        .chain(entries.iter().map(|e| Listing {
            path: &e.path,
            format: e.format,
            description: &e.description,
        }))
        .collect();
    let entry =
        |l: &Listing<'static>, bytes: Vec<u8>| Entry::new(l.path, bytes, l.format, l.description);
    let readme = entry(&README, readme_text(&listings).into_bytes());
    let ro_crate = entry(
        &RO_CRATE,
        serde_json::to_vec_pretty(&ro_crate_json(&listings, created))?,
    );
    Ok([readme, ro_crate].into_iter().chain(entries).collect())
}

fn readme_text(entries: &[Listing]) -> String {
    let mut text = String::from(
        "# Duello session\n\n\
         Saved by duello (https://github.com/mlund/duello). Reopen the whole \
         `.duello.zip` with File ▸ Open session in duello-web, or pass it to \
         `duello nbody --scan`. Keep it zipped: some browsers (Safari's \"Open \
         safe files after downloading\") unzip downloads, and the folder can't \
         be reopened.\n\n\
         JSON, YAML and structure files are what duello reads back. CSV and XYZ \
         files are copies for spreadsheets and viewers; editing them changes \
         nothing on reopening.\n\n\
         Units: lengths in Å, energies in kT (k = Boltzmann constant, T as \
         in the scan settings), charges in e, concentrations in mol/L.\n\n\
         | File | Contents |\n|---|---|\n",
    );
    for entry in entries {
        text += &format!("| `{}` | {} |\n", entry.path, entry.description);
    }
    text
}

fn ro_crate_json(entries: &[Listing], created: &str) -> Value {
    let files = entries.iter().filter(|e| e.path != RO_CRATE.path);
    let parts: Vec<Value> = files.clone().map(|e| json!({ "@id": e.path })).collect();
    let mut graph = vec![
        json!({
            "@id": RO_CRATE.path,
            "@type": "CreativeWork",
            "conformsTo": { "@id": "https://w3id.org/ro/crate/1.1" },
            "about": { "@id": "./" },
        }),
        json!({
            "@id": "./",
            "@type": "Dataset",
            "name": "Duello session",
            "description": "Inputs and results of a duello protein–protein interaction session",
            "datePublished": created,
            "hasPart": parts,
        }),
    ];
    graph.extend(files.map(|e| {
        json!({
            "@id": e.path,
            "@type": "File",
            "name": e.path.rsplit('/').next(),
            "encodingFormat": e.format,
            "description": e.description,
        })
    }));
    json!({ "@context": "https://w3id.org/ro/crate/1.1/context", "@graph": graph })
}
