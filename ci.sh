#!/usr/bin/env bash
set -euo pipefail

cargo fmt --all --check

clippy() {
    echo "clippy $*" >&2
    cargo clippy "$@" -- -D clippy::pedantic -D warnings
}

clippy
clippy --no-default-features

if [ "$(uname -s)" = "Linux" ]; then
  ext=rename2,io-uring
  clippy --no-default-features --features $ext
else
  ext=
fi

clippy --features fm-bench,sa-bench,sa-packed-bench,diff,dhat-heap,$ext
