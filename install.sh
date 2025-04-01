#!/bin/bash

# Save script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set work directory
WORKDIR="/tmp/labs"
mkdir -p $WORKDIR
cd $WORKDIR

# Clone ClipSeg repository
git clone https://github.com/timojl/clipseg
cd clipseg
rm -rf .git/

# Download and extract weights
wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip
unzip -d weights -j weights.zip

# Move clipseg to the script directory
cd ..
mv clipseg "$SCRIPT_DIR"
