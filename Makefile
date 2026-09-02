# ==============================================================================
#  Makefile for ECDLP Solvers and Benchmarks
# ==============================================================================

# --- Compiler Configuration ---
CC      = cc
CFLAGS  = -O3 -Wall -Wextra -Wno-unused-function -Wno-unused-result

# --- secp256k1 Paths ---
# Point this to your local cloned repository directory
SECP256K1_DIR = ../secp256k1
SECP256K1_SRC = $(SECP256K1_DIR)/src

# --- Include and Linker Flags ---
# Includes the internal src/ directory and the public API include/ directory
INCLUDES = -I/usr/local/include -I$(SECP256K1_SRC) -I$(SECP256K1_DIR)/include

# LDFLAGS: Static linking against libsecp256k1 by pointing to the root-level hidden .libs folder where libsecp256k1.a resides
LDFLAGS  = -L$(SECP256K1_DIR)/.libs
## Or dynamic linking against libsecp256k1
# LDFLAGS  =

# Libraries to link against
LIBS       = -lsecp256k1 -lpthread
LIBS_BENCH = -lsecp256k1

# --- Target Binaries ---
TARGETS = bsgs \
		  bsgs_zaddsub \
		  bsgs_zaddc \
          fastecdlp_treemon \
          fastecdlp_parallel \
          fastecdlp_jacobian \
          bench_field

# ==============================================================================
#  Build Rules
# ==============================================================================

.PHONY: all clean

# Default target: builds everything
all: $(TARGETS)

# 1. Our complete solver
bsgs: bsgs_dlp_benchmark_cached.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) $(LIBS)

# 2. Our complete solver using zaddsub
bsgs_zaddsub: bsgs_dlp_benchmark_cached_zaddsub.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) $(LIBS)

# 3. Our complete solver using zaddc
bsgs_zaddc: bsgs_dlp_benchmark_cached_zaddc.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) $(LIBS)

# 4. FastECDLP faithful (Tang et al.)
fastecdlp_treemon: fastecdlp_treemon.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) $(LIBS)

# 5. FastECDLP + parallel Phase 1+2
fastecdlp_parallel: fastecdlp_parallel.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) $(LIBS)

# 6. FastECDLP + Jacobian loop (no T₂)
fastecdlp_jacobian: fastecdlp_jacobian.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) $(LIBS)

# 7. Field microbenchmark (does not require pthread)
bench_field: bench_field.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) $(LIBS_BENCH)

# --- Clean Utility ---
clean:
	@echo "Cleaning up build artifacts..."
	rm -f $(TARGETS)