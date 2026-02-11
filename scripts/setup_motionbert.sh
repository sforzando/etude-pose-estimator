#!/bin/bash
# MotionBERT setup script
#
# This script sets up MotionBERT for 3D pose lifting.
# It clones the MotionBERT repository and downloads the checkpoint file.
#
# Usage:
#   bash scripts/setup_motionbert.sh
#   or
#   task setup-motionbert

set -e  # Exit on error

echo "================================================================"
echo "MotionBERT Setup Script"
echo "================================================================"
echo ""

# Define paths
MODELS_DIR="models"
MOTIONBERT_DIR="$MODELS_DIR/MotionBERT"
CHECKPOINT_DIR="$MODELS_DIR/motionbert"
CHECKPOINT_FILE="$CHECKPOINT_DIR/checkpoint.pth.tar"
MOTIONBERT_REPO="https://github.com/Walter0807/MotionBERT.git"

# Create directories
echo "📁 Creating directories..."
mkdir -p "$MODELS_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Step 1: Clone MotionBERT repository
if [ -d "$MOTIONBERT_DIR" ]; then
    echo "✓ MotionBERT repository already exists at $MOTIONBERT_DIR"
else
    echo "📥 Cloning MotionBERT repository..."
    git clone "$MOTIONBERT_REPO" "$MOTIONBERT_DIR"
    echo "✓ Repository cloned successfully"
fi

# Step 2: Download checkpoint
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "✓ Checkpoint already exists at $CHECKPOINT_FILE"
else
    echo "📥 Downloading MotionBERT checkpoint..."
    echo "   This may take a few minutes (~2GB file)..."

    # Note: The actual checkpoint URL needs to be verified
    # This is a placeholder - update with actual URL from MotionBERT releases
    CHECKPOINT_URL="https://github.com/Walter0807/MotionBERT/releases/download/v1.0/checkpoint.pth.tar"

    # Try to download with curl or wget
    if command -v curl &> /dev/null; then
        curl -L -o "$CHECKPOINT_FILE" "$CHECKPOINT_URL"
    elif command -v wget &> /dev/null; then
        wget -O "$CHECKPOINT_FILE" "$CHECKPOINT_URL"
    else
        echo "❌ Error: Neither curl nor wget found. Please install one of them."
        exit 1
    fi

    echo "✓ Checkpoint downloaded successfully"
fi

# Step 3: Create symbolic link (optional)
SYMLINK_PATH="src/etude_pose_estimator/motionbert"
if [ -L "$SYMLINK_PATH" ] || [ -d "$SYMLINK_PATH" ]; then
    echo "✓ MotionBERT library link already exists"
else
    echo "🔗 Creating symbolic link for MotionBERT library..."
    ln -s "../../$MOTIONBERT_DIR/lib" "$SYMLINK_PATH"
    echo "✓ Symbolic link created"
fi

# Step 4: Verify installation
echo ""
echo "🔍 Verifying installation..."
if [ -d "$MOTIONBERT_DIR" ] && [ -f "$CHECKPOINT_FILE" ]; then
    echo "✅ MotionBERT setup complete!"
    echo ""
    echo "Installation summary:"
    echo "  - Repository: $MOTIONBERT_DIR"
    echo "  - Checkpoint: $CHECKPOINT_FILE"
    echo "  - Library link: $SYMLINK_PATH"
    echo ""
    echo "Next steps:"
    echo "  1. Update MOTIONBERT_MODEL_PATH in .envrc if needed"
    echo "  2. Run 'task dev' to start the development server"
    echo "  3. Test pose detection with sample images"
else
    echo "❌ Setup verification failed!"
    echo "Please check the error messages above."
    exit 1
fi

echo ""
echo "================================================================"
echo "✨ Setup complete!"
echo "================================================================"
