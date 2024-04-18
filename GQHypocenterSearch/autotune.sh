#!/bin/bash

# Specify the values of BLOCK_HYPOCS and TILE
BLOCK_HYPOCS_VALUES=(32 64 128 224 256 264 280 324 360 384 442 512 1024)
TILE_VALUES=(1 2 3 4 5 6 7 8 10 12)

filename="autotune_results.csv"

# Remove the existing file if it exists
rm -f "$filename"
touch "$filename"
echo "BLOCK_HYPOCS,TILE,Best_PPS">"$filename"

# Create the build directory if it doesn't exist
mkdir -p build

# CD into the build directory
cd build

# Loop over BLOCK_HYPOCS values
for BLOCK_HYPOCS in "${BLOCK_HYPOCS_VALUES[@]}"; do
    # Loop over TILE values
    for TILE in "${TILE_VALUES[@]}"; do
        # Run cmake with specified BLOCK_HYPOCS and TILE values
        cmake -DBLOCK_HYPOCS=$BLOCK_HYPOCS -DTILE=$TILE ..

        # Compile the project using make
        make

        # Optionally, run your project here if needed
        ./bin/gq_test

        # Optionally, clean the build directory after each iteration
        # make clean
    done
done
