//! A pair closer than its table start is an overlap, so exact Yukawa can't
//! pull opposite charges through a soft short-range wall.

mod common;

use common::{duello, example, run, workdir};

/// (R, F/kT) rows of a `--pmf` file.
fn read_pmf(path: &std::path::Path) -> Vec<(f64, f64)> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let mut cols = l.split_whitespace().map(|c| c.parse::<f64>().unwrap());
            (cols.next().unwrap(), cols.next().unwrap())
        })
        .collect()
}

// Invariant: overlaps stay finite and repulsive; the only well is Yukawa's at contact.
#[test]
fn soft_wall_ion_pair_does_not_collapse() {
    let dir = workdir("overlap_soft_ions");
    run(duello(&dir)
        .args(["scan", "-1"])
        .arg(example("soft-ions/PP.xyz"))
        .arg("-2")
        .arg(example("soft-ions/NP.xyz"))
        .arg("--top")
        .arg(example("soft-ions/topology.yaml"))
        .args(["-M", "0.02", "--rmin", "0.1", "--rmax", "12", "--dr", "0.1"])
        .args(["--max-ndiv", "0", "--pmf", "pmf.dat"]));
    let pmf = read_pmf(&dir.join("pmf.dat"));
    assert!(pmf.iter().all(|(_, f)| f.is_finite()), "{pmf:?}");
    let (r_min, f_min) = pmf
        .iter()
        .copied()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    // Yukawa at 1 Å is about −7 kT; collapsing to 0.1 Å would reach −140 kT.
    assert!(f_min > -10.0, "well of {f_min:.1} kT at {r_min} Å");
    assert!(r_min >= 1.0, "well inside the table start, at {r_min} Å");
}
