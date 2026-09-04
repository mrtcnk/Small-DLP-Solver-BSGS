#!/usr/bin/env bash
# ==============================================================================
#  benchmark.sh — Reproducible benchmark for BSGS Small ECDLP Solver
#
#  Runs all four implementations with the same seeds and reports results
#  in a format suitable for inclusion in the paper.
#
#  Usage:
#    ./benchmark.sh                    # auto-detect thread count
#    ./benchmark.sh --threads 10       # use 10 threads
#    ./benchmark.sh --threads 1        # single-threaded
#    ./benchmark.sh --threads 24       # Muhammad's workstation
#    ./benchmark.sh --skip-build       # skip make (binaries already built)
#
#  Requirements:
#    - All binaries built via: make
#    - Baby tables pre-built for l1=31 (run once: ./bsgs_zaddsub 54 31 1 1 512 12345)
#
# ==============================================================================

set -euo pipefail

# --- Configuration ---
# Auto-detect logical CPU count; can be overridden with --threads
THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)
SEEDS="12345 99999 54321"   # multiple seeds for stability
TRIALS_SMALL=10             # 52/54-bit
TRIALS_LARGE=3              # 58/63/64-bit
WINDOW=512
L1=31
SKIP_BUILD=0
LOG_FILE="benchmark_$(date +%Y%m%d_%H%M%S).txt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --threads)   THREADS="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --- Helper functions ---
log() { echo "$1" | tee -a "$LOG_FILE"; }
separator() { log "$(printf '=%.0s' {1..70})"; }

# --- Header ---
separator
log "BSGS Small ECDLP Solver — Benchmark Report"
log "Date     : $(date)"
log "Machine  : $(uname -srm)"
log "CPU      : $(sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo 'unknown')"
log "Cores    : $(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 'unknown')"
log "Threads  : $THREADS"
log "Seeds    : $SEEDS"
log "Window   : $WINDOW"
log "l1       : $L1"
log "Log file : $LOG_FILE"
separator

# --- Build ---
if [[ $SKIP_BUILD -eq 0 ]]; then
    log ""
    log "Building all implementations..."
    make 2>&1 | tee -a "$LOG_FILE"
    log "Build complete."
fi

# --- Check binaries ---
BINARIES="fastecdlp_treemon fastecdlp_parallel bsgs bsgs_zaddsub"
for bin in $BINARIES; do
    if [[ ! -x "./$bin" ]]; then
        log "ERROR: ./$bin not found. Run 'make' first."
        exit 1
    fi
done
log ""
log "All binaries found."

# --- Run benchmarks ---

run_benchmark() {
    local label=$1
    local binary=$2
    local bits=$3
    local trials=$4
    local extra=${5:-""}

    log ""
    log "--- $label ---"
    if [[ -n "$extra" ]]; then
        # bsgs variants need window argument
        for seed in $SEEDS; do
            log "  seed=$seed:"
            ./"$binary" "$bits" "$L1" "$trials" "$THREADS" "$WINDOW" "$seed" 2>&1 | \
                grep -E "Average per solve|Solved correctly" | \
                sed 's/^/    /' | tee -a "$LOG_FILE"
        done
    else
        # fastecdlp variants: no window argument
        for seed in $SEEDS; do
            log "  seed=$seed:"
            ./"$binary" "$bits" "$L1" "$trials" "$THREADS" "$seed" 2>&1 | \
                grep -E "Average per solve|Solved correctly|Total time" | \
                sed 's/^/    /' | tee -a "$LOG_FILE"
        done
    fi
}

run_benchmark_worst_case() {
    local label=$1
    local binary=$2
    local bits=$3
    local extra=${5:-""}

    log ""
    log "--- $label (worst case: m = 2^$bits - 1) ---"
    if [[ -n "$extra" ]]; then
        ./"$binary" "$bits" "$L1" 1 "$THREADS" "$WINDOW" 0 2>&1 | \
            grep -E "Average per solve|Solved correctly" | \
            sed 's/^/    /' | tee -a "$LOG_FILE"
    else
        ./"$binary" "$bits" "$L1" 1 "$THREADS" 0 2>&1 | \
            grep -E "Average per solve|Solved correctly|Total time" | \
            sed 's/^/    /' | tee -a "$LOG_FILE"
    fi
}

# 54-bit
separator
log "54-bit benchmarks (l1=$L1, W=$WINDOW, threads=$THREADS, trials=$TRIALS_SMALL)"
separator
run_benchmark "FastECDLP TreeMon (Tang et al.)"  fastecdlp_treemon  54 $TRIALS_SMALL ""
run_benchmark "FastECDLP Parallel (our Ph.1+2)"  fastecdlp_parallel 54 $TRIALS_SMALL ""
run_benchmark "BSGS (windowed batch inversion)"  bsgs               54 $TRIALS_SMALL "window"
run_benchmark "BSGS zaddsub (Co-Z arithmetic)"   bsgs_zaddsub       54 $TRIALS_SMALL "window"

separator
log ""
log "54-bit worst-case benchmarks (l1=$L1, W=$WINDOW, threads=$THREADS, m=2^54-1)"
separator
run_benchmark_worst_case "FastECDLP TreeMon (Tang et al.)"  fastecdlp_treemon  54 1 ""
run_benchmark_worst_case "FastECDLP Parallel (our Ph.1+2)"  fastecdlp_parallel 54 1 ""
run_benchmark_worst_case "BSGS (windowed batch inversion)"  bsgs               54 1 "window"
run_benchmark_worst_case "BSGS zaddsub (Co-Z arithmetic)"   bsgs_zaddsub       54 1 "window"

# 58-bit
separator
log ""
log "58-bit benchmarks (l1=$L1, W=$WINDOW, threads=$THREADS, trials=$TRIALS_LARGE)"
separator
run_benchmark "FastECDLP TreeMon (Tang et al.)"  fastecdlp_treemon  58 $TRIALS_LARGE ""
run_benchmark "FastECDLP Parallel (our Ph.1+2)"  fastecdlp_parallel 58 $TRIALS_LARGE ""
run_benchmark "BSGS (windowed batch inversion)"  bsgs               58 $TRIALS_LARGE "window"
run_benchmark "BSGS zaddsub (Co-Z arithmetic)"   bsgs_zaddsub       58 $TRIALS_LARGE "window"

separator
log ""
log "58-bit worst-case benchmarks (l1=$L1, W=$WINDOW, threads=$THREADS, m=2^58-1)"
separator
run_benchmark_worst_case "FastECDLP TreeMon (Tang et al.)"  fastecdlp_treemon  58 1 ""
run_benchmark_worst_case "FastECDLP Parallel (our Ph.1+2)"  fastecdlp_parallel 58 1 ""
run_benchmark_worst_case "BSGS (windowed batch inversion)"  bsgs               58 1 "window"
run_benchmark_worst_case "BSGS zaddsub (Co-Z arithmetic)"   bsgs_zaddsub       58 1 "window"

# 63-bit
separator
log ""
log "63-bit benchmarks (l1=$L1, W=$WINDOW, threads=$THREADS, trials=$TRIALS_LARGE)"
log "Note: FastECDLP variants infeasible (T2 = 87.5 GB)"
separator
run_benchmark "BSGS (windowed batch inversion)"  bsgs         63 $TRIALS_LARGE "window"
run_benchmark "BSGS zaddsub (Co-Z arithmetic)"   bsgs_zaddsub 63 $TRIALS_LARGE "window"

separator
log ""
log "63-bit worst-case benchmarks (l1=$L1, W=$WINDOW, threads=$THREADS, m=2^63-1)"
separator
run_benchmark_worst_case "BSGS (windowed batch inversion)"  bsgs         63 1 "window"
run_benchmark_worst_case "BSGS zaddsub (Co-Z arithmetic)"   bsgs_zaddsub 63 1 "window"

# 64-bit
separator
log ""
log "64-bit benchmarks (l1=$L1, W=$WINDOW, threads=$THREADS, trials=$TRIALS_LARGE)"
log "Note: FastECDLP variants infeasible (T2 = 175 GB)"
separator
run_benchmark "BSGS (windowed batch inversion)"  bsgs         64 $TRIALS_LARGE "window"
run_benchmark "BSGS zaddsub (Co-Z arithmetic)"   bsgs_zaddsub 64 $TRIALS_LARGE "window"

separator
log ""
log "64-bit worst-case benchmarks (l1=$L1, W=$WINDOW, threads=$THREADS, m=2^64-1)"
separator
run_benchmark_worst_case "BSGS (windowed batch inversion)"  bsgs         64 1 "window"
run_benchmark_worst_case "BSGS zaddsub (Co-Z arithmetic)"   bsgs_zaddsub 64 1 "window"

# --- Summary ---
separator
log ""
log "Benchmark complete. Full results saved to: $LOG_FILE"
separator