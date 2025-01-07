#!/bin/sh

example=$(basename "$0" .sh)

cd examples/${example}
RUST_LOG="Info" cargo run --release \
    -- scan \
    -1 cppm-p18.xyz \
    -2 cppm-p18.xyz \
    --icotable \
    --rmin 37 --rmax 121 --dr 0.5 \
    --top topology.yaml \
    --resolution 0.8 \
    --cutoff 1000 \
    --molarity 0.005 \
    --temperature 298.15
