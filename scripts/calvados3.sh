#!/bin/sh

example=$(basename "$0" .sh)
cd examples/${example}

RUST_LOG="Info" cargo run --release \
    -- scan \
    -1 4lzt.xyz \
    -2 4lzt.xyz \
    --rmin 24 --rmax 50 --dr 1.0 \
    --top topology.yaml \
    --resolution 1.0 \
    --cutoff 1000 \
    --molarity 0.050 \
    --temperature 298.15
