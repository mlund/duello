- Performance benchmarks should be based on:
~~~
cd examples/calvados3/
duello scan -1 2CDS.xyz -2 2CDS.xyz --rmin 26 --rmax 60 --dr 0.5 --top topology.yaml --molarity 0.02
~~~
  Timings from before the topology set each pair's spline range (`--cutoff 150`
  splined out to 150 Å) are not comparable.
  With a GPU this runs the grid backend (1.0 Å maps); pin `--backend gpu` for the
  exact-pair GPU kernel, e.g. to compare with baselines from before the grid.
  The `--savescan` archive below is then grid-filled too.
  Add `--savescan lys.duello.zip` to benchmark the many-body engine on it (from
  the same directory; cargo runs tests from the package root, hence `$PWD`):
~~~
DUELLO_BENCH_ARCHIVE=$PWD/lys.duello.zip cargo test --release -p duello --lib step8_benchmark -- --ignored --nocapture
~~~
  The grid backend's accuracy (B₂ vs `--backend gpu`) and speed on the same
  system, plus its contact energies on chymotrypsinogen (7KTZ), from the
  package root:
~~~
cargo test --release -p duello --lib grid_scan -- --ignored --nocapture
~~~
- Avoid lenghty Rust functions
- Reduce nested code with helper functions
- Prefer *why* over *what* in code comments
- For git commits, be brief and don't add (co)-authorship
