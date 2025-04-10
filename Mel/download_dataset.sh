#!/bin/bash

# Define variables
DATA_DIR="data"
TAR_NAME="genres.tar.gz"
TAR_URL="http://opihi.cs.uvic.ca/sound/genres.tar.gz"
TAR_PATH="$DATA_DIR/$TAR_NAME"
EXTRACT_DIR="$DATA_DIR/genres"

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Download the tar.gz file if it doesn't exist
if [ ! -f "$TAR_PATH" ]; then
    echo "📥 Downloading GTZAN dataset..."
    wget -O "$TAR_PATH" "$TAR_URL"
    echo "✅ Download complete."
else
    echo "📁 Archive already exists. Skipping download."
fi

# Extract the tar.gz file if the target directory doesn't exist
if [ ! -d "$EXTRACT_DIR" ]; then
    echo "🗜️ Extracting GTZAN dataset..."
    
    if tar -xzf "$TAR_PATH" -C "$DATA_DIR"; then
        echo "✅ Extraction complete."

        # Remove archive after successful extraction
        echo "🧹 Removing archive file..."
        rm "$TAR_PATH"
        echo "✅ Archive removed."
    else
        echo "❌ Extraction failed. Keeping archive file for troubleshooting."
        echo "💡 Please ensure you have 'tar' installed."
    fi
else
    echo "📂 GTZAN dataset already extracted. Skipping extraction."
fi
