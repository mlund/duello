- Performance benchmarks should be based on:
~~~
cd examples/calvados3/
duello scan -1 4lzt.xyz -2 4lzt.xyz --rmin 26 --rmax 60 --dr 0.5 --top topology.yaml --resolution 0.5 --cutoff 150 --molarity 0.02`
~~~
- Avoid lenghty Rust functions
- Reduce nested code with helper functions
- Prefer *why* over *what* in code comments
- For git commits, be brief and don't add (co)-authorship
