"""QC admin view.

Live numbers come only from the API's SQLite log. The MVTec table, when the
results file is present, is the offline benchmark already reported in the
README. It is not this line's defect rate.

The savings panel is a calculator. The throughput field is filled from
results/throughput.json when that file is on disk. That file is a CPU
measurement of the encoder, not a measured dollar saving.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defectclip.inspection_log import InspectionLog
from defectclip.savings import estimate_labor_delta

DATA_DIR = Path(os.environ.get("QC_DATA_DIR", ROOT / "var" / "qc"))
DB_PATH = DATA_DIR / "inspections.db"
RESULTS = ROOT / "results"


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def main() -> None:
    st.set_page_config(page_title="QC dashboard", layout="wide")
    st.title("Inspection log")
    st.caption(f"Reading {DB_PATH}. Empty until something calls POST /predict.")

    log = InspectionLog(DB_PATH)
    summary = log.summary()
    columns = st.columns(5)
    columns[0].metric("Inspections", summary["n"])
    columns[1].metric("Defect decisions", summary["n_defect"])
    columns[2].metric("Pass decisions", summary["n_pass"])
    columns[3].metric("Unscored", summary["n_unscored"])
    columns[4].metric("Alerts", summary["n_alerts"])

    st.subheader("Defect rate by category")
    st.caption(
        "Defects divided by scored inspections (pass + defect). Unscored rows are left out."
    )
    scored = [row for row in summary["by_category"] if row["n_scored"]]
    if not scored:
        st.info("No scored inspections yet.")
    else:
        st.bar_chart({row["category"]: row["defect_rate"] for row in scored})
        st.dataframe(scored, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Recent inspections")
        recent = log.recent_inspections(30)
        if recent:
            st.dataframe(recent, hide_index=True)
        else:
            st.write("None yet.")
    with right:
        st.subheader("Recent alerts")
        alerts = log.recent_alerts(30)
        if alerts:
            st.dataframe(
                [
                    {
                        "created_at": row["created_at"],
                        "category": row["category"],
                        "score": row["score"],
                        "threshold": row["threshold"],
                        "image_name": row["image_name"],
                        "channels": ", ".join(row["channels"]),
                        "errors": "; ".join(row["errors"]),
                    }
                    for row in alerts
                ],
                hide_index=True,
            )
        else:
            st.write("None yet. Alerts fire only when a decision is defect.")

    st.subheader("Offline MVTec benchmark")
    st.caption(
        "These AUROCs are from results/zeroshot.json. They are not live defect rates."
    )
    zeroshot = _load_json(RESULTS / "zeroshot.json")
    if zeroshot is None:
        st.write("results/zeroshot.json is not in this image.")
    else:
        rows = [
            {"category": category, "image_auroc": values["auroc"]}
            for category, values in zeroshot["per_category"].items()
        ]
        st.metric("Mean image AUROC", f"{zeroshot['mean_auroc']:.3f}")
        st.dataframe(rows, hide_index=True)

    st.subheader("Labor-time calculator")
    st.warning(
        "Illustration only. Type your line rate, manual time, and wage. "
        "The result is arithmetic. It is not a measured saving, and it ignores misses, false alarms, and rework."
    )
    throughput = _load_json(RESULTS / "throughput.json")
    measured = None
    if throughput is not None:
        measured = throughput["online_predict_after_text_prototypes_cached"]["images_per_second"]
        online = throughput["online_predict_after_text_prototypes_cached"]
        st.caption(
            f"Default throughput is the online encode measured on {throughput['measured_on']} "
            f"({throughput['machine']}, {throughput['model']}, {throughput['device']}): "
            f"{online['n_images']} images in {online['seconds']:.3f}s = {measured:.4f} images/second, "
            "after text prototypes were already built. Batch-16 on the same machine was "
            f"{throughput['batch_16']['images_per_second']:.4f} images/second."
        )
    parts = st.number_input("Parts per clock hour", min_value=0.0, value=0.0, step=10.0)
    manual = st.number_input("Manual seconds spent looking at one part", min_value=0.0, value=0.0, step=0.5)
    wage = st.number_input("Labor cost, USD per hour", min_value=0.0, value=0.0, step=1.0)
    rate = st.number_input(
        "Model images per second",
        min_value=0.0,
        value=float(measured or 0.0),
        step=0.1,
        help="Pre-filled from the CPU measurement when results/throughput.json is available.",
    )
    if rate <= 0:
        st.info("Enter a model throughput above zero. The shipped default is the measured CPU rate.")
    else:
        estimate = estimate_labor_delta(parts, manual, wage, rate)
        st.json(estimate)


if __name__ == "__main__":
    main()
