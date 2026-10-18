"""Condition completed split evidences without loading maps or posterior arrays.

Usage: normalize_physical_evidence.py SUMMARY.json MORPHOLOGY.json OUTPUT.json
The original file is never overwritten. The output records the box evidence,
conditional piece evidences, physical piece weights and unchanged posteriors.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"ktzobs"))
from scripts.physical_prior import inputs_from_metadata, condition_split_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("morphology", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.resolve() in (args.summary.resolve(), args.morphology.resolve()):
        parser.error("Use a separate output file to preserve the source audit")
    data = json.loads(args.summary.read_text())
    metadata = json.loads(args.morphology.read_text())
    inputs = inputs_from_metadata(metadata, (data["_pieces"][0][0], data["_pieces"][-1][1]))
    result = condition_split_summary(data, inputs)
    result["_physical_prior_inputs"] = inputs
    args.output.write_text(json.dumps(result, indent=1)+"\n")
    z0 = result["L0"]["logZ"]
    for name in ("L0", "L1", "L1n", "L2"):
        row = result[name]
        print(f"{name}: F={row['retained_prior_mass']:.10f}, lnZ={row['logZ']:.8f}, Delta lnZ={row['logZ']-z0:+.8f}")


if __name__ == "__main__":
    main()
