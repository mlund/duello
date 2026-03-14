#!/bin/sh

example=$(basename "$0" .sh)

cd examples/${example}
RUST_LOG="Info" cargo run --release -- \
    scan \
    -1 cppm-p18.xyz \
    -2 cppm-p18.xyz \
    --rmin 38.0 --rmax 100 --dr 0.5 \
    --top topology.yaml \
    --cutoff 2000 \
    --molarity 0.005 \
    --max-ndiv 2 \
    --savetable P18-P18-5mM.dat.gz
    #--xtcfile traj.xtc
