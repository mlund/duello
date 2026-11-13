- Performance benchmarks should be based on:
~~~
cd examples/calvados3/
duello scan -1 2CDS.xyz -2 2CDS.xyz --rmin 26 --rmax 60 --dr 0.5 --top topology.yaml --cutoff 150 --molarity 0.02
~~~
  Add `--savescan lys.duello.zip` to benchmark the many-body engine on it (from
  the same directory; cargo runs tests from the package root, hence `$PWD`):
~~~
DUELLO_BENCH_ARCHIVE=$PWD/lys.duello.zip cargo test --release -p duello --lib step8_benchmark -- --ignored --nocapture
~~~
- Avoid lenghty Rust functions
- Reduce nested code with helper functions
- Prefer *why* over *what* in code comments
- For git commits, be brief and don't add (co)-authorship
