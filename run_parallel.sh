#!/bin/bash

# -------- Configuration --------
TRIALS=5
THREADS_LIST="1 2 4 8"
BITS_LIST="40 41 42 43 44"
L1_START=20
L1_END=23
BIN=./bsgs_mt

# -------- Run sweep --------

echo "==============================================="
echo "BSGS Multithread Benchmark"
echo "Trials per config: $TRIALS"
echo "Threads tested: $THREADS_LIST"
echo "==============================================="
echo

for BITS in $BITS_LIST
do
    for L1 in $(seq $L1_START $L1_END)
    do
        if [ "$L1" -lt "$BITS" ]; then

            for THREADS in $THREADS_LIST
            do
                echo "Running: bits=$BITS l1=$L1 trials=$TRIALS threads=$THREADS"
                $BIN $BITS $L1 $TRIALS $THREADS
                echo "--------------------------------------------------"
            done

        fi
    done
done

echo
echo "Benchmark completed."