#!/bin/sh

example=$(basename "$0" .sh)

cd examples/${example}
RUST_LOG="Info" cargo run --release -- \
    scan \
    -1 cppm-p18.xyz \
    -2 cppm-p18.xyz \
    --rmin 39.0 --rmax 60 --dr 1.0 \
    --top topology.yaml \
    --resolution 0.8 \
    --cutoff 60 \
    --molarity 0.05 \
    --backend simd \
    #--savetable table.dat.gz \
    #--xtcfile traj.xtc
