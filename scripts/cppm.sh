#!/bin/sh

example=$(basename "$0" .sh)

cd examples/${example}
RUST_LOG="Info" cargo run --release -- \
    scan \
    -1 cppm-p18.xyz \
    -2 cppm-p18.xyz \
    --rmin 38.0 --rmax 70 --dr 2.0 \
    --top topology.yaml \
    --cutoff 2000 \
    --molarity 0.05 \
    --max-ndiv 2 \
    --savetable P18-P18-50mM.dat.gz

RUST_LOG="Info" cargo run --release -- \
    scan \
    -1 cppm-p00.xyz \
    -2 cppm-p18.xyz \
    --rmin 38.0 --rmax 70 --dr 1.0 \
    --top topology.yaml \
    --cutoff 2000 \
    --molarity 0.05 \
    --max-ndiv 2 \
    --savetable P00-P18-50mM.dat.gz
