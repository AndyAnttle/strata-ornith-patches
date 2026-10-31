#!/usr/bin/env bash
# Apply the Ornith engine patches to a Strata source tree.
#
#   ./apply.sh /path/to/Strata          (the tree that holds src/, include/, CMakeLists.txt)
#
# Order matters: 01 adds the dequant include that 03 extends.
set -euo pipefail
TREE="${1:-}"
if [ -z "$TREE" ] || [ ! -d "$TREE/src/qwen35" ]; then
    echo "usage: $0 /path/to/Strata" >&2
    exit 2
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
for p in "$HERE"/patches/*.diff; do
    echo "== $(basename "$p")"
    ( cd "$TREE" && patch -p1 --forward --batch --no-backup-if-mismatch < "$p" ) || {
        echo "failed: $(basename "$p") -- the tree may be a newer Strata than the patches were cut for" >&2
        exit 1
    }
done
echo "patched: $TREE"
echo "build: cmake -S \"$TREE\" -B \"$TREE/build-qwen35\" -G Ninja -DCMAKE_BUILD_TYPE=Release \\"
echo "         -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=OFF \\"
echo "         -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_GGML_DIR=<the pinned llama.cpp>"
echo "       ninja -C \"$TREE/build-qwen35\" strata strata_kq"
