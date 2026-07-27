# AGT Polygon Fixer

A guided desktop app (Streamlit) that wraps the WV3 polygon alignment / QA /
topology pipeline into a step-by-step workflow.

## Run it

- **Quick:** double-click `run_agt_fixer_silent.vbs` (server runs hidden, opens
  in your browser).
- **Install shortcuts:** run `install_shortcuts.bat` once for Desktop + Start
  Menu entries.
- **From a terminal:**
  ```
  C:\Users\phila\venvs\agtfix314\Scripts\python.exe -m streamlit run agt_fix_app.py
  ```

## The workflow (tabs)

1. **WV3 data** — point to the WorldView-3 raster (GeoTIFF/JP2). Read from disk,
   not uploaded (rasters are large).
2. **Polygons** — point to the polygon layer (GeoPackage/Shapefile); only
   features inside the raster footprint are loaded.
3. **Configure & run** — tick which corrections to apply (remove duplicates /
   correct shift / per-polygon refinement / fix overlaps) and set the keep-drop
   thresholds, then **Run**.
4. **Review** — class counts, an overlay map, a filterable table, and a button to
   open **FiftyOne** for visual keep/drop triage on image chips.
5. **Topology** — planarization result (overlap removed, residual = 0 means no
   double-counted area).
6. **Export** — write the GeoPackage (`agt_qa` full record, `agt_clean` planar
   deliverable, `agt_original` pre-shift geometry).

## Architecture

- The pipeline runs in-process in the `agtfix314` env (Python 3.14:
  rasterio / onnxruntime / geopandas). Stage code is reused unchanged from
  `../Notebooks/fix_agt_polygons.py`.
- The **Review** tab shells out to the separate `fiftyone311` env (Python 3.11)
  — `app_pipeline.prepare_fiftyone_package` chips the raster + writes a manifest
  in the geo env, then `launch_fiftyone.py` builds the dataset and serves the
  FiftyOne app, which the page embeds.
- Override the FiftyOne interpreter with the `AGTFIX_FIFTYONE_PY` env var.

## Files

| File | Role |
|------|------|
| `agt_fix_app.py` | Streamlit UI (the tabs) |
| `app_pipeline.py` | pipeline entry point + FiftyOne package prep |
| `launch_fiftyone.py` | builds/launches FiftyOne (runs in the 3.11 env) |
| `run_agt_fixer.bat` / `.vbs` | launchers |
| `install_shortcuts.bat` | Desktop/Start-Menu shortcuts |
| `installer/` | Inno Setup script + build guide for a distributable `.exe` |

For a distributable installer for other machines, see
[`installer/BUILD_INSTALLER.md`](installer/BUILD_INSTALLER.md).
