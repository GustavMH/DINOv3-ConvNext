#!/usr/bin/env bash
# Usage: bash scripts/run_pilot.sh prepare | distill sat|web | probe
set -euo pipefail
cd "$(dirname "$0")/.."
export LIGHTLY_TRAIN_EVENTS_DISABLED=1
crops=data/crops
out=${OUT_DIR:-results}

prepare() { [[ -f $1/meta.json ]] || python -u -m satdistill.prepare "${@:2}" --out "$1"; }

probe() {
    python -u -m satdistill.probe --name chesapeake "$1" --train $crops/ches_md_train \
        --test $crops/ches_md_test $crops/ches_ny_test --out-dir "$out"
    python -u -m satdistill.probe --name loveda "$1" --train $crops/loveda_train --test $crops/loveda_val \
        --out-dir "$out"
}

case ${1:-} in
    prepare)
        prepare $crops/distill chesapeake --splits {de,pa,va,wv}-{train,val,test} --max-per-tile "${MAX_PER_TILE:-200}"
        prepare $crops/ches_md_train chesapeake --splits md-train --labels --max-per-tile 30 --max-crops 3000
        prepare $crops/ches_md_test chesapeake --splits md-test --labels --max-per-tile 75
        prepare $crops/ches_ny_test chesapeake --splits ny-test --labels --max-per-tile 75
        prepare $crops/loveda_train loveda --split train --labels --max-crops 3000
        prepare $crops/loveda_val loveda --split val --labels --max-crops 1500
        ;;
    distill)
        python -u -m satdistill.distill --teacher "${2:?usage: $0 distill sat|web}" --method "${METHOD:-v3}" \
            --data $crops/distill --epochs "${EPOCHS:-10}" --batch-size "${BATCH:-64}" --workers "${WORKERS:-8}" \
            --out-dir "$out"
        ;;
    probe)
        for model in convnext_tiny.dinov3_lvd1689m vit_large_patch16_dinov3.lvd1689m vit_large_patch16_dinov3.sat493m; do
            probe --model=$model
        done
        for ckpt in "$out"/student_*.pt; do
            [[ -f $ckpt ]] && probe --ckpt="$ckpt"
        done
        python -m satdistill.summarize --out-dir "$out"
        ;;
    *)
        echo "usage: $0 prepare | distill sat|web | probe" >&2
        exit 1
        ;;
esac