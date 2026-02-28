#!/bin/bash

TRIALS=5

for BITS in 40 41 42 43 44
do
    for L1 in $(seq 18 23)
    do
        # Skip invalid splits
        if [ "$L1" -lt "$BITS" ]; then
            echo "Running: bits=$BITS l1=$L1 trials=$TRIALS"
            ./bsgs_dlp $BITS $L1 $TRIALS
            echo "--------------------------------------------------"
        fi
    done
done
