#!/usr/bin/env bash
# Download every model in configs/models.tsv to /scratch1 at its pinned revision
# and verify the checksums. About 25 GB. Rerunning resumes and skips finished
# files, so if a login-node limit kills it, run it again.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo/configs/carc.env"

pb_login_limits

hf="$PB_ENV/bin/hf"
if [[ ! -x "$hf" ]]; then
    echo "ERROR: $hf not found; run scripts/setup_env.sh first" >&2
    exit 1
fi

while IFS=$'\t' read -r name repo_id revision role; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    echo "== $name ($role): $repo_id@$revision"
    "$hf" download "$repo_id" --revision "$revision" --local-dir "$PB_MODELS/$name" --max-workers 4 </dev/null
    "$hf" cache verify "$repo_id" --revision "$revision" --local-dir "$PB_MODELS/$name" </dev/null
done < "$PB_MODEL_LIST"

echo "All models downloaded and verified under $PB_MODELS"
