#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bucket=s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/lcmcvpr2019/cvpr_chesapeake_landcover

case ${1:-} in
    chesapeake)
        mkdir -p data/chesapeake
        aws s3 sync --no-sign-request --region us-west-2 "$bucket/" data/chesapeake \
            --exclude "*" --include "*_naip-new.tif" --include "*_lc.tif" --include "spatial_index.geojson"
        du -sh data/chesapeake
        ;;
    loveda)
        mkdir -p data/loveda
        for split in Train Val; do
            if [[ ! -d data/loveda/$split ]]; then
                wget -c -O "data/loveda/$split.zip" "https://zenodo.org/records/5706578/files/$split.zip?download=1"
                unzip -q "data/loveda/$split.zip" -d data/loveda
                rm "data/loveda/$split.zip"
            fi
        done
        du -sh data/loveda
        ;;
    *)
        echo "usage: $0 chesapeake|loveda" >&2
        exit 1
        ;;
esac
