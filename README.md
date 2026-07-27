# AGT Polygon Fixer

Streamlit desktop app for aligning, QA-ing and cleaning AGT building/property
polygons against WorldView-3 imagery (Property Tax Recovery Project, Luanda,
Angola). Same algorithms as the ArcGIS Pro edition of this pipeline, wrapped
in a plain-Python guided workflow: WV3 data -> Polygons -> Configure & run ->
Review -> Topology -> Export.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app/agt_fix_app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (see below).
2. Go to https://share.streamlit.io, sign in with GitHub, click **New app**.
3. Pick this repository/branch, set **Main file path** to `app/agt_fix_app.py`.
4. Deploy.

## The building-detection model

<!-- TODO: filled in once the model-hosting approach is decided -->
`buildseg_segformer_b2.onnx` (~110 MB) is not committed directly (GitHub's
file-size limit is 100 MB). See `docs/MODEL.md` for how it is provided at
runtime.

## Notes for the cloud deployment

- The **Review** tab's FiftyOne integration shells out to a separate local
  Python environment and will not work on Streamlit Community Cloud; every
  other tab (align, score, classify, planarize, topology, export) runs
  in-process and works there.
- Rasters and polygon layers are read from a path you type in, not uploaded
  through the browser (imagery is large) -- on Streamlit Cloud that means the
  data must already be reachable from the container (e.g. downloaded at
  startup, or the deployment adapted to accept file uploads for smaller AOIs).
