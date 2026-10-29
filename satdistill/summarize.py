"""Collect the probe results into results/summary.csv and print a table (mIoU per test set and model).

Example: python -m satdistill.summarize
"""
import argparse
import csv
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    rows = []
    for path in sorted(glob.glob(os.path.join(args.out_dir, "probe_*.json"))):
        with open(path) as f:
            r = json.load(f)
        for test, values in r["results"].items():
            rows.append({"probe": r["probe"], "test": test, "model": r["model"], "mIoU": round(values["mIoU"], 4),
                         "pixel_accuracy": round(values["pixel_accuracy"], 4)})
    if not rows:
        print("no probe results in", args.out_dir)
        return
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tests = sorted({(r["probe"], r["test"]) for r in rows if r["test"] != "train"})
    models = sorted({r["model"] for r in rows})
    table = {(r["probe"], r["test"], r["model"]): r["mIoU"] for r in rows}
    width = max(len(m) for m in models)
    print("mIoU".ljust(width) + "".join("  %18s" % t[1][:18] for t in tests))
    for m in models:
        print(m.ljust(width) + "".join("  %18s" % (("%.4f" % table[t + (m,)]) if t + (m,) in table else "-")
                                       for t in tests))
    print("Saved", os.path.join(args.out_dir, "summary.csv"))


if __name__ == "__main__":
    main()
