#!/usr/bin/env bash
# Download the public corpus into data/raw. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/nist data/raw/py
MCP_COMMIT="${MCP_COMMIT:-54b91cdb8f21825ce586c452704f319d49866ef8}"

# 1. Python 3.13 documentation, HTML build (PSF-2.0 / docs license)
[ -f data/raw/pydocs.zip ] || curl -sSfL -o data/raw/pydocs.zip https://docs.python.org/3.13/archives/python-3.13-docs-html.zip
python3 -c "import zipfile;zipfile.ZipFile('data/raw/pydocs.zip').extractall('data/raw/py')"

# 2. MCP specification repo (spec: Apache-2.0 / MIT during relicensing, docs CC-BY-4.0)
if [ ! -d data/raw/mcp_repo ]; then
  git clone -q https://github.com/modelcontextprotocol/modelcontextprotocol.git data/raw/mcp_repo
fi
git -C data/raw/mcp_repo checkout -q "$MCP_COMMIT"

# 3. NIST publications (US government works, not subject to copyright in the US)
for f in ai/NIST.AI.100-1.pdf ai/NIST.AI.600-1.pdf ai/NIST.AI.100-2e2025.pdf CSWP/NIST.CSWP.29.pdf \
         SpecialPublications/NIST.SP.800-218.pdf SpecialPublications/NIST.SP.800-207.pdf \
         SpecialPublications/NIST.SP.800-63-3.pdf SpecialPublications/NIST.SP.800-61r3.pdf \
         SpecialPublications/NIST.SP.800-37r2.pdf SpecialPublications/NIST.SP.800-88r1.pdf; do
  out="data/raw/nist/$(basename "$f")"
  [ -f "$out" ] || curl -sSfL -o "$out" "https://nvlpubs.nist.gov/nistpubs/$f"
done
sha256sum -c data/sha256_raw.txt || echo "WARNING: checksums differ from the run in results/ (Python 3.13 docs are rebuilt on each patch release)"
echo "corpus downloaded"
