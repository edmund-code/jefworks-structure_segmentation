# label_glomeruli_app.py
# Streamlit UI to label candidate glomerulus boxes on an NDPI WSI.
# - Two-column layout: image (left) + controls (right)
# - Fixed-height scrollable image panel so controls stay visible
# - Reads NDPI with OpenSlide at an appropriate pyramid level (fast)
# - Saves labels incrementally to out/box_labels.csv

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import openslide

# -------------------------
# CONFIG: set your paths
# -------------------------
NDPI_PATH = "/Users/edmundtsou/Desktop/JEFworks/jefworks-structure_segmentation/data/lab_kidney_data_ndpi/OTS-24-22043 - 2024-08-28 15.08.37.ndpi"

# Your candidate boxes CSV (e.g., region_split_max3.csv)
BOXES_CSV = "/Users/edmundtsou/Desktop/JEFworks/jefworks-structure_segmentation/out/region_split_max3.csv"

OUTDIR = "/Users/edmundtsou/Desktop/JEFworks/jefworks-structure_segmentation/out"
os.makedirs(OUTDIR, exist_ok=True)
LABELS_CSV = os.path.join(OUTDIR, "box_labels.csv")

# -------------------------
# UI/Crop settings
# -------------------------
DEFAULT_PAD_FRAC = 0.08      # context around bbox
PANEL_HEIGHT_PX = 520        # scrollable image panel height
TARGET_VIEW_PX = 1100        # max dimension for displayed crop (controls how zoomed-out it looks)
BATCH_SAVE_EVERY = 1         # save every label

# -------------------------
# Helpers
# -------------------------
def clamp_box(x0, y0, x1, y1, W, H):
    x0 = max(0, min(W, int(round(x0))))
    y0 = max(0, min(H, int(round(y0))))
    x1 = max(0, min(W, int(round(x1))))
    y1 = max(0, min(H, int(round(y1))))
    if x1 <= x0:
        x1 = min(W, x0 + 1)
    if y1 <= y0:
        y1 = min(H, y0 + 1)
    return x0, y0, x1, y1

def expand_box(x0, y0, x1, y1, pad_frac):
    w = x1 - x0
    h = y1 - y0
    px = int(round(w * pad_frac))
    py = int(round(h * pad_frac))
    return x0 - px, y0 - py, x1 + px, y1 + py

@st.cache_resource
def load_slide(path):
    return openslide.OpenSlide(path)

@st.cache_data
def load_boxes(csv_path):
    df = pd.read_csv(csv_path)
    required = {"x0", "y0", "x1", "y1"}
    if not required.issubset(df.columns):
        raise ValueError(f"Boxes CSV must contain columns {required}. Got: {df.columns.tolist()}")

    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})

    df["id"] = df["id"].astype(int)
    # make sure numeric coords
    for c in ["x0", "y0", "x1", "y1"]:
        df[c] = df[c].astype(float)

    return df

def load_or_init_labels(path):
    if os.path.exists(path):
        lab = pd.read_csv(path)
        if "id" in lab.columns:
            lab["id"] = lab["id"].astype(int)
        # ensure columns exist
        for c in ["label", "multi", "notes", "ts"]:
            if c not in lab.columns:
                lab[c] = np.nan
        return lab[["id","label","multi","notes","ts"]]
    return pd.DataFrame(columns=["id","label","multi","notes","ts"])

def upsert_label(df, rec):
    # rec: dict with id,label,multi,notes,ts
    if len(df) == 0:
        return pd.DataFrame([rec])
    m = df["id"] == rec["id"]
    if m.any():
        df.loc[m, ["label","multi","notes","ts"]] = [rec["label"], rec["multi"], rec["notes"], rec["ts"]]
        return df
    return pd.concat([df, pd.DataFrame([rec])], ignore_index=True)

def label_for_id(labels_df, _id):
    m = labels_df["id"] == _id
    if m.any():
        r = labels_df[m].iloc[0]
        lbl = r["label"] if pd.notna(r["label"]) else None
        multi = bool(r["multi"]) if pd.notna(r["multi"]) else False
        notes = r["notes"] if pd.notna(r["notes"]) else ""
        return str(lbl) if lbl is not None else None, multi, str(notes)
    return None, False, ""

def read_crop(slide, x0, y0, x1, y1, target_view_px=1100):
    """Read a region using the best pyramid level so its largest dimension ~ target_view_px."""
    W0, H0 = slide.dimensions
    x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W0, H0)
    w = x1 - x0
    h = y1 - y0

    want_down = max(w, h) / float(target_view_px)
    level = slide.get_best_level_for_downsample(want_down)
    down = float(slide.level_downsamples[level])

    x0_l = int(x0 / down)
    y0_l = int(y0 / down)
    w_l  = max(1, int(w / down))
    h_l  = max(1, int(h / down))

    img = slide.read_region((x0_l, y0_l), level, (w_l, h_l)).convert("RGB")

    # final safety resize (keeps UI consistent)
    max_dim = max(img.width, img.height)
    if max_dim > target_view_px:
        s = target_view_px / max_dim
        img = img.resize((max(1, int(img.width*s)), max(1, int(img.height*s))), Image.BICUBIC)

    meta = {"level": level, "down": down, "w0": w, "h0": h, "wl": w_l, "hl": h_l}
    return img, meta

# -------------------------
# Page setup
# -------------------------
st.set_page_config(layout="wide")
st.title("Glomerulus Box Labeling (WSI NDPI)")

# CSS for scrollable image panel + slightly tighter spacing
st.markdown(
    f"""
    <style>
    .imgbox {{
        height: {PANEL_HEIGHT_PX}px;
        overflow-y: auto;
        border: 1px solid rgba(49,51,63,.2);
        border-radius: 10px;
        padding: 10px;
        background: rgba(250,250,250,.65);
    }}
    .block-container {{ padding-top: 1rem; padding-bottom: 1.5rem; }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("View")
    only_unlabeled = st.checkbox("Show unlabeled only", value=True)
    pad_frac = st.slider("Pad around box", 0.0, 0.5, DEFAULT_PAD_FRAC, 0.01)
    target_px = st.slider("Zoom-out (max crop px)", 400, 2200, TARGET_VIEW_PX, 100)
    panel_h = st.slider("Image panel height", 300, 900, PANEL_HEIGHT_PX, 20)

    st.caption("If controls are still not visible, reduce panel height or increase zoom-out.")

    sort_mode = st.selectbox("Order", ["id", "heat_value (desc)"], index=0)

# allow dynamic panel height without restarting
st.markdown(
    f"""
    <style>
    .imgbox {{
        height: {panel_h}px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Load data
# -------------------------
slide = load_slide(NDPI_PATH)
W0, H0 = slide.dimensions

boxes = load_boxes(BOXES_CSV)
labels = load_or_init_labels(LABELS_CSV)

# merge label columns
boxes2 = boxes.copy()
if len(labels) > 0:
    boxes2 = boxes2.merge(labels[["id","label","multi","notes"]], on="id", how="left")
else:
    boxes2["label"] = np.nan
    boxes2["multi"] = np.nan
    boxes2["notes"] = np.nan

if sort_mode == "heat_value (desc)" and "heat_value" in boxes2.columns:
    boxes2 = boxes2.sort_values("heat_value", ascending=False)
else:
    boxes2 = boxes2.sort_values("id", ascending=True)

if only_unlabeled:
    boxes_view = boxes2[boxes2["label"].isna()].copy()
else:
    boxes_view = boxes2.copy()

if len(boxes_view) == 0:
    st.success("No boxes to show (all labeled under current filter).")
    st.stop()

# -------------------------
# Session state: index
# -------------------------
if "idx" not in st.session_state:
    st.session_state.idx = 0

st.session_state.idx = max(0, min(len(boxes_view) - 1, st.session_state.idx))

# Jump controls
with st.sidebar:
    st.divider()
    st.subheader("Navigate")
    st.write(f"Boxes in view: **{len(boxes_view)}**")
    jump_id = st.number_input("Jump to box id", value=int(boxes_view.iloc[0]["id"]), step=1)

    colj1, colj2 = st.columns(2)
    with colj1:
        if st.button("Jump"):
            ids = boxes_view["id"].values
            where = np.where(ids == int(jump_id))[0]
            if len(where) > 0:
                st.session_state.idx = int(where[0])
            else:
                st.session_state.idx = int(np.argmin(np.abs(ids - int(jump_id))))
            st.rerun()
    with colj2:
        if st.button("Reset to first"):
            st.session_state.idx = 0
            st.rerun()

# current row
row = boxes_view.iloc[st.session_state.idx]
box_id = int(row["id"])

# bbox coords
x0, y0, x1, y1 = float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"])
x0, y0, x1, y1 = expand_box(x0, y0, x1, y1, pad_frac)
x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W0, H0)

# read crop at appropriate level
img, meta = read_crop(slide, x0, y0, x1, y1, target_view_px=target_px)

# label state for this id
cur_label, cur_multi, cur_notes = label_for_id(labels, box_id)
if cur_label is None:
    cur_label = "unsure"

# -------------------------
# Main layout: 2 columns
# -------------------------
left, right = st.columns([2.2, 1.2], gap="large")

with left:
    # header info
    info = []
    if "parent_id" in row: info.append(f"parent_id={int(row['parent_id'])}")
    if "peak_rank" in row: info.append(f"peak_rank={int(row['peak_rank'])}")
    if "heat_value" in row and pd.notna(row["heat_value"]): info.append(f"heat={float(row['heat_value']):.3f}")
    st.markdown(f"**Box ID:** {box_id} &nbsp;&nbsp; **Index:** {st.session_state.idx+1}/{len(boxes_view)}")
    if info:
        st.caption(" • ".join(info))

    # scrollable image panel
    st.markdown('<div class="imgbox">', unsafe_allow_html=True)
    st.image(img)  # no use_container_width; keeps panel predictable
    st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        f"level={meta['level']} down={meta['down']:.2f} | "
        f"level0_box={int(x1-x0)}x{int(y1-y0)} px | "
        f"display={img.width}x{img.height} px"
    )

with right:
    st.subheader("Label")
    label = st.radio(
        "Class",
        options=["glomerulus", "not_glomerulus", "unsure"],
        index=0 if cur_label == "glomerulus" else 1 if cur_label == "not_glomerulus" else 2,
        horizontal=False
    )
    multi = st.checkbox("Multiple glomeruli in crop", value=cur_multi)
    notes = st.text_input("Notes (optional)", value=cur_notes, placeholder="partial glom, sclerotic, vessel, fold...")

    st.divider()

    def save_current():
        nonlocal_labels = load_or_init_labels(LABELS_CSV)  # robust: reload latest
        rec = {
            "id": box_id,
            "label": label,
            "multi": bool(multi),
            "notes": notes,
            "ts": int(time.time())
        }
        nonlocal_labels = upsert_label(nonlocal_labels, rec)
        nonlocal_labels.to_csv(LABELS_CSV, index=False)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save"):
            save_current()
            st.success("Saved.")
    with c2:
        if st.button("Save + Next"):
            save_current()
            st.session_state.idx = min(len(boxes_view) - 1, st.session_state.idx + 1)
            st.rerun()

    c3, c4 = st.columns(2)
    with c3:
        if st.button("Prev"):
            st.session_state.idx = max(0, st.session_state.idx - 1)
            st.rerun()
    with c4:
        if st.button("Next"):
            st.session_state.idx = min(len(boxes_view) - 1, st.session_state.idx + 1)
            st.rerun()

    st.divider()

    st.caption("Files")
    st.code(f"Boxes: {BOXES_CSV}\nLabels: {LABELS_CSV}", language="text")
