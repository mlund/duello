
#!/bin/sh

cd ../examples/cppm/

molarities="0.001"
temperatures="280"

for t in $temperatures
do
	for mol in $molarities
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
        --pmf pmf_${t}_${mol}.dat \
        --log log_${t}_${mol}.log
    done
done


