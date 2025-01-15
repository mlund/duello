
#!/bin/sh

cd ../examples/cppm/

molalities="0.001 0.0015 0.002"
temperatures="280"

#(seq 0.0025 0.0025 0.01 | sed 's/0*$//')


for t in {311..330}
do
    for mol in $(seq 0.0175 0.0025 0.15 | sed 's/0*$//')

    do
    RUST_LOG="Info" cargo run --release \
        -- scan \
        -1 cppm-p18.xyz \
        -2 cppm-p18.xyz \
        --icotable \
        --rmin 37 --rmax 121 --dr 0.5 \
        --top topology.yaml \
        --resolution 0.8 \
        --cutoff 1000 \
        --molarity $mol \
        --temperature $t \
        --pmf pmf/pmf_${t}_${mol}.dat \
        --log log/log_${t}_${mol}.log
    done
done




