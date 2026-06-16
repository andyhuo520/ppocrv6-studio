#!/usr/bin/env bash
# =============================================================================
# Download PP-OCRv6 ONNX models
# Fetches pre-converted ONNX files from the project's GitHub Releases.
# Usage: bash scripts/download_models.sh [tiny|small|medium|all]
# =============================================================================
set -e

REPO="andyhu/ppocrv6-studio"
BASE="https://github.com/${REPO}/releases/latest/download"
TARGET="${1:-all}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

download_model() {
    local name=$1     # e.g. "tiny"
    local dir=$2      # e.g. "ppocrv6_onnx"
    local tarball="${name}.tar.gz"

    info "Downloading ${name} model…"
    curl -fL --progress-bar "${BASE}/${tarball}" -o "/tmp/${tarball}" || \
        error "Download failed. Check https://github.com/${REPO}/releases for available assets."

    mkdir -p "${dir}"
    tar -xzf "/tmp/${tarball}" --strip-components=1 -C "${dir}"
    rm "/tmp/${tarball}"
    info "Extracted → ./${dir}/"
}

case "$TARGET" in
    tiny)
        download_model tiny  ppocrv6_onnx
        ;;
    small)
        download_model small ppocrv6_small_onnx
        ;;
    medium)
        download_model medium ppocrv6_medium_onnx
        ;;
    all)
        download_model tiny   ppocrv6_onnx
        download_model small  ppocrv6_small_onnx
        download_model medium ppocrv6_medium_onnx
        ;;
    *)
        echo "Usage: bash scripts/download_models.sh [tiny|small|medium|all]"
        exit 1
        ;;
esac

info "Done. Expected directory layout:"
echo "  ppocrv6_onnx/          ← Tiny  (1.5 MB official params)"
echo "  ppocrv6_small_onnx/    ← Small (7.7 MB)"
echo "  ppocrv6_medium_onnx/   ← Medium (34.5 MB)"
echo ""
info "Start the studio: python webapp/server.py"
