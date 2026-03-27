#!/bin/sh

example=$(basename "$0" .sh)
cd examples/${example}

MODE=${1:-all}  # "all" (default), "scan", or "diffusion"

MOLARITIES="0.10 0.95 1.47"
LABELS="100mM 950mM 1470mM"

set -- $LABELS
for mol in $MOLARITIES; do
    label=$1; shift
    table="lysozyme-${label}-pH45.dat.gz"

    if [ "$MODE" = "all" ] || [ "$MODE" = "scan" ]; then
        echo "=== Scanning ${label} (molarity=${mol} M) ==="
        cargo run --release \
            -- scan \
            -1 4lzt.xyz \
            -2 4lzt.xyz \
            --rmin 24 --rmax 60 --dr 1.0 \
            --top topology.yaml \
            --max-ndiv 2 \
            --cutoff 2000 \
            --molarity $mol \
            --temperature 293.15 \
            --savetable $table \
            --pmf pmf_${label}.dat \
            > scan_${label}.log 2>&1
    fi

    if [ "$MODE" = "all" ] || [ "$MODE" = "diffusion" ]; then
        echo "=== Diffusion analysis ${label} ==="
        cargo run --release \
            -- diffusion $table \
            -T 293.15 \
            -o diffusion_${label}.csv \
            --cmin 0.035e-3 --cmax 1.398e-3 --cn 20 \
            --cell-output cell_diffusion_${label}.csv \
            > diffusion_${label}.log 2>&1
    fi
done
