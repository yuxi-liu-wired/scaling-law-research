#!/bin/bash

# Check if a filename has been provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

FILENAME=$1

# Ensure the file exists
if [ ! -f "$FILENAME" ]; then
    echo "File does not exist: $FILENAME"
    exit 1
fi

# Get the original file size
original_size=$(stat -c%s "$FILENAME")

# Function to print compression ratio
print_compression_ratio() {
    compressed_size=$(stat -c%s "$1")
    ratio=$(echo "scale=8; $original_size / $compressed_size" | bc)
    echo "$2 compression ratio: $ratio:1"
}

# xz compression
xz -9eck "$FILENAME" > "$FILENAME.xz"
print_compression_ratio "$FILENAME.xz" "xz"

# 7za compression
7za a -t7z -mx=9 "$FILENAME.7z" "$FILENAME" > /dev/null
print_compression_ratio "$FILENAME.7z" "7za"

# zstd compression
zstd --ultra -22 -k "$FILENAME" -o "$FILENAME.zst"
print_compression_ratio "$FILENAME.zst" "zstd"

# Brotli compression
brotli -Zk "$FILENAME" -o "$FILENAME.br"
print_compression_ratio "$FILENAME.br" "Brotli"

# PAQ compression (example with paq8l, adjust as needed)
paq8px -8ael "$FILENAME" "$FILENAME.paq8l" # Uncomment if paq8l is available
print_compression_ratio "$FILENAME.paq8l" "PAQ8l" # Uncomment if paq8l is available

# Clean up compressed files (optional, uncomment if desired)
rm -f "$FILENAME.xz" "$FILENAME.7z" "$FILENAME.zst" "$FILENAME.br" "$FILENAME.paq8l"

