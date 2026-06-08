#!/usr/bin/env bash
# Build the Deka technical white paper: Markdown -> Pandoc -> Tectonic -> PDF.
set -euo pipefail
cd "$(dirname "$0")"

pandoc whitepaper.md \
  --metadata-file=metadata.yaml \
  --pdf-engine=tectonic \
  --toc \
  -o whitepaper.pdf

echo "Built whitepaper.pdf ($(du -h whitepaper.pdf | cut -f1))"
