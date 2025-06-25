#!/bin/sh

example=$(basename "$0" .sh)

cd examples/${example}
RUST_LOG="Debug" cargo run --release -- \
    scan \
    -1 cppm-p18.xyz \
    -2 cppm-p18.xyz \
    --rmin 40.0 --rmax 150 --dr 1.0 \
    --top topology.yaml \
    --resolution 0.8 \
    --cutoff 1000 \
    --molarity 0.005 \
    #--savetable table.dat.gz \
    #--xtcfile traj.xtc
    --temperature 330
