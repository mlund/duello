#!/bin/sh

example=$(basename "$0" .sh)
cd examples/${example}

RUST_LOG="Debug" cargo run --release \
    -- scan \
    -1 4lzt.xyz \
    -2 4lzt.xyz \
    --rmin 24 --rmax 80 --dr 0.5 \
    --top topology.yaml \
    --resolution 0.7 \
    --cutoff 1000 \
    --molarity 0.05 \
    --temperature 298.15
