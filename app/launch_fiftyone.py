"""Build a FiftyOne dataset from a prepared chip manifest and launch the app.

Runs in the FiftyOne 3.11 venv (separate from the app env). Consumes the
manifest.json written by app_pipeline.prepare_fiftyone_package (chips + normalised
polylines), so it needs only fiftyone. Polylines are coloured by qa_class.

Usage: python launch_fiftyone.py --manifest <path> [--name agt_review] [--port 5151]
"""
import argparse
import json
import os


def build(manifest_path, name):
    import fiftyone as fo

    with open(manifest_path) as fh:
        man = json.load(fh)

    if name in fo.list_datasets():
        fo.delete_dataset(name)
    ds = fo.Dataset(name, persistent=True)

    samples = []
    for s in man["samples"]:
        polylines = []
        for pl in s["polylines"]:
            pts = [[(float(x), float(y)) for x, y in ring] for ring in pl["points"]]
            if not pts or len(pts[0]) < 4:      # guard degenerate rings (app crash)
                continue
            polylines.append(fo.Polyline(label=pl["label"], points=pts,
                                         closed=True, filled=False))
        sample = fo.Sample(filepath=s["filepath"])
        sample["polygons"] = fo.Polylines(polylines=polylines)
        samples.append(sample)
    ds.add_samples(samples)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--name", default="agt_review")
    ap.add_argument("--port", type=int, default=5151)
    args = ap.parse_args()

    import fiftyone as fo
    ds = build(args.manifest, args.name)
    print(f"Dataset '{args.name}': {len(ds)} chips, launching on port {args.port}")
    session = fo.launch_app(ds, address="127.0.0.1", port=args.port)
    session.wait(-1)                            # keep the process (and app) alive


if __name__ == "__main__":
    main()
