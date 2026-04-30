#!/bin/bash
# benchmark.sh — measure BSGS solve times across bit sizes and l1 splits
# Usage: ./benchmark.sh [threads]
# Output is printed to stdout and also saved to benchmark_results.txt

THREADS=${1:-10}
TRIALS=3
BINARY=./bsgs
OUTFILE=benchmark_results.txt

# Bit sizes and l1 ranges to sweep
BITS_LIST=(44 46 48 50 52)

# l1 range: keep l2 = bits - l1 in [10..25] for reasonable search time
# l1_min = bits - 25, l1_max = bits - 10
# Also cap l1 at 28 (table ~4GB) and floor at 16

echo "=======================================================" | tee $OUTFILE
echo "BSGS Benchmark — secp256k1, 64-bit key, Jacobian loop" | tee -a $OUTFILE
echo "Threads: $THREADS | Trials: $TRIALS"                    | tee -a $OUTFILE
echo "=======================================================" | tee -a $OUTFILE
echo ""                                                        | tee -a $OUTFILE

for BITS in "${BITS_LIST[@]}"; do
    echo "--- bits=$BITS ---" | tee -a $OUTFILE

    L1_MIN=$(( BITS - 25 ))
    L1_MAX=$(( BITS - 10 ))

    # Clamp
    [ "$L1_MIN" -lt 16 ] && L1_MIN=16
    [ "$L1_MAX" -gt 28 ] && L1_MAX=28

    for L1 in $(seq $L1_MIN $L1_MAX); do
        L2=$(( BITS - L1 ))

        # Skip if l2 too small or l1 >= bits
        if [ "$L2" -lt 8 ] || [ "$L1" -ge "$BITS" ]; then
            continue
        fi

        echo "  bits=$BITS l1=$L1 l2=$L2 threads=$THREADS trials=$TRIALS" | tee -a $OUTFILE
        $BINARY $BITS $L1 $TRIALS $THREADS 2>&1 | grep -E "Solved|Average|Table|Init" | tee -a $OUTFILE
        echo "" | tee -a $OUTFILE
    done

    echo "" | tee -a $OUTFILE
done

echo "Results saved to $OUTFILE"