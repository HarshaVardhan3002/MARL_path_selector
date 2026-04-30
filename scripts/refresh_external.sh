#!/usr/bin/env bash
# Regenerate the gitignored external/ directory by re-cloning upstream repos
# at their pinned SHAs. Safe to re-run.
#
# Note: mpquic-on-scion-ipc contains files with colons in their names; on
# Windows we extract a Windows-safe subset via per-file `git show` (see
# JUSTIFICATIONS.md J-008). Linux/macOS users get a normal checkout.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$PROJECT_ROOT/external"
mkdir -p "$EXT"

clone_pinned() {
  local url="$1" sha="$2" name
  name="$(basename "$url" .git)"
  local dst="$EXT/$name"
  if [ ! -d "$dst/.git" ]; then
    git clone --no-checkout "$url" "$dst"
  fi
  ( cd "$dst" && git fetch --depth 1 origin "$sha" && git checkout "$sha" 2>/dev/null || true )
  echo "$name @ $(cd "$dst" && git rev-parse HEAD)"
}

clone_pinned https://github.com/keshvadi/Path-Aware-Multipath-Instability.git \
  294c07979c1d58128ac3ec80e18288388d5bf60f
clone_pinned https://github.com/Keshvadi/mpquic-on-scion-ipc.git \
  362ad60885dabeb2a067d3dae058c506ce5f092a
clone_pinned https://github.com/keshvadi/mpcc-dynamics.git \
  5d0595cee65480a213cf347c40e03bfbcc38858b

# Windows-safe subset extraction for mpquic-on-scion-ipc (see J-008).
SCIONPATHML_REPO="$EXT/mpquic-on-scion-ipc"
SCIONPATHML_DST="$EXT/scionpathml_extracted"
mkdir -p "$SCIONPATHML_DST"
( cd "$SCIONPATHML_REPO" && \
  for f in README.md LICENSE \
    AnalysisScripts/README.md AnalysisScripts/analyze_bw.py \
    AnalysisScripts/analyze_comparer.py AnalysisScripts/analyze_prober.py \
    AnalysisScripts/analyze_sp-mp-bw.py AnalysisScripts/analyze_sp-mp-prober.py \
    AnalysisScripts/analyze_traceroute.py \
    AnalysisResults/Lab1/Analysis/comparer_analysis.txt \
    AnalysisResults/Lab1/Analysis/prober_analysis.txt \
    AnalysisResults/Lab1/Analysis/sp_bw_analysis.txt \
    AnalysisResults/Lab1/Analysis/sp_mp_bw_comparison.txt \
    AnalysisResults/Lab1/Analysis/sp_mp_prober_comparison.txt \
    AnalysisResults/Lab1/Data/raw_data.zip \
    AnalysisResults/Lab1/Data/selected_paths.json \
    AnalysisResults/Lab2/Analysis/comparer_analysis.txt \
    AnalysisResults/Lab2/Analysis/prober_analysis.txt \
    AnalysisResults/Lab2/Analysis/sp_bw_analysis.txt \
    AnalysisResults/Lab2/Analysis/sp_mp_bw_comparison.txt \
    AnalysisResults/Lab2/Analysis/sp_mp_prober_comparison.txt \
    AnalysisResults/Lab2/Data/raw_data.zip \
    AnalysisResults/Lab2/Data/selected_paths.json \
    AnalysisResults/Lab3/Analysis/comparer_analysis.txt \
    AnalysisResults/Lab3/Analysis/prober_analysis.txt \
    AnalysisResults/Lab3/Analysis/sp_bw_analysis.txt \
    AnalysisResults/Lab3/Analysis/sp_mp_bw_comparison.txt \
    AnalysisResults/Lab3/Analysis/sp_mp_prober_comparison.txt \
    AnalysisResults/Lab3/Data/raw_data.zip \
    AnalysisResults/Lab3/Data/selected_paths.json
  do
    mkdir -p "$SCIONPATHML_DST/$(dirname "$f")"
    git show "HEAD:$f" > "$SCIONPATHML_DST/$f"
  done )
echo "Extracted ScionPathML subset to $SCIONPATHML_DST"
