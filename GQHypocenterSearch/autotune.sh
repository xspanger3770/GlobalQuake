#!/bin/bash

# Specify the values of BLOCK_HYPOCS and TILE
BLOCK_HYPOCS_VALUES=(256 384 512 768)
TILE_VALUES=(4 5 6 7 8 9 10 11 12 13 14)

#BLOCK_HYPOCS_VALUES=(32)
#TILE_VALUES=(5)

DEFAULT_TESTS=12
TESTS=${1:-$DEFAULT_TESTS}

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
        cmake -DBLOCK_HYPOCS=$BLOCK_HYPOCS -DTILE=$TILE -DTESTS=$TESTS ..

        # Compile the project using make
        make

        # Optionally, run your project here if needed
        ./bin/gq_test

        # Optionally, clean the build directory after each iteration
        # make clean
    done
done

cd ..

highest_value=$(awk -F';' 'NR > 1 {print $3}' "$filename" | sort -n -r | head -n 1)

# Use grep to find the entire row with the highest value in the third column
row=$(grep "$highest_value" "$filename")

echo "Best solution:"
echo "$row"
echo "$row">best_solution.txt
