#!/usr/bin/env bash
set -euo pipefail

root="${1:-.}"
cd "$root"

fail=0

report_failure() {
  local title="$1"
  local file="$2"
  echo "::error::$title"
  sed -n '1,120p' "$file"
  fail=1
}

large_files="$(mktemp)"
find . -path './.git' -prune -o -type f -size +50M -print >"$large_files"
if [[ -s "$large_files" ]]; then
  report_failure "Files larger than 50 MB are not allowed in the source repo" "$large_files"
fi

model_artifacts="$(mktemp)"
find . -path './.git' -prune -o -type f \( \
  -name '*.safetensors' -o \
  -name '*.gguf' -o \
  -name '*.mlx' -o \
  -name '*.bin' -o \
  -name '*.npz' -o \
  -name '*.npy' \
\) -print >"$model_artifacts"
if [[ -s "$model_artifacts" ]]; then
  report_failure "Model artifacts do not belong in Git" "$model_artifacts"
fi

workspace_residue="$(mktemp)"
find . -path './.git' -prune -o \( \
  -path './models' -o \
  -path './outputs' -o \
  -path './REFERENCES:TOOLS' -o \
  -path './DEEP RESEARCH HANDOFF' -o \
  -path './IDE RESEARCH' -o \
  -path './ProAgent Details' -o \
  -path './.venv' -o \
  -name '.webui_secret_key' \
\) -print >"$workspace_residue"
if [[ -s "$workspace_residue" ]]; then
  report_failure "Private workspace residue found" "$workspace_residue"
fi

bad_names="$(mktemp)"
find . -path './.git' -prune -o -type f -print \
  | awk 'index($0, " ") || index($0, ":") { print }' >"$bad_names"
if [[ -s "$bad_names" ]]; then
  report_failure "Tracked/public paths must not contain spaces or colons" "$bad_names"
fi

secret_matches="$(mktemp)"
rg --hidden --line-number --glob '!.git/**' --glob '!dist/**' --glob '!build/**' \
  'TOKEN|SECRET|PASSWORD|API_KEY|webui_secret|gho_|hf_' . >"$secret_matches" || true

filtered_secrets="$(mktemp)"
awk '
  /scripts\/hygiene_scan\.sh/ { next }
  /\.gitignore:/ { next }
  /PREFIX_DIVERGENCE_AT_TOKEN/ { next }
  /MTPLX_LATE_DEPTH_SWITCH_AFTER_TOKENS/ { next }
  /hf_path/ { next }
  /mtplx\/artifacts\.py:.*(_hf_|hf_hub_|huggingface_hub|HF_TOKEN|HUGGING_FACE_HUB_TOKEN)/ { next }
  /mtplx\/hf_loader\.py:.*(hf_|_hf_|HF_TOKEN|HUGGING_FACE_HUB_TOKEN)/ { next }
  /mtplx\/cli\.py:.*(hf_loader|DEFAULT_HF_MODEL_ID)/ { next }
  /mtplx\/commands\/public\.py:.*(hf_loader|hf_cache_report|huggingface)/ { next }
  /tests\/test_artifacts\.py:.*(_hf_|test_hf|hf_)/ { next }
  /tests\/test_hf_loader\.py:.*(hf_|HF_TOKEN|HUGGING_FACE_HUB_TOKEN)/ { next }
  /tests\/test_no_mlx_imports\.py:.*test_run_reports_uncached_hf_model/ { next }
  { print }
' "$secret_matches" >"$filtered_secrets"

if [[ -s "$filtered_secrets" ]]; then
  report_failure "Potential secret patterns found" "$filtered_secrets"
fi

exit "$fail"
