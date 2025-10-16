import os
import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

from src.nutrition import load_nutrition_db, fuzzy_lookup, get_nutrition_for, scale_per_serving

st.set_page_config(page_title="Food Nutrition Detector", page_icon="🍽️", layout="wide")

# ---- Paths ----
DB_PATH = os.path.join("data", "nutrition_db.csv")
SAMPLE_IMG = os.path.join("assets", "sample.jpg")

@st.cache_resource(show_spinner=True)
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, processor, device

@st.cache_data(show_spinner=False)
def load_labels_and_df():
    df = load_nutrition_db(DB_PATH)
    labels = df["food_name"].tolist()
    return labels, df

def clip_zero_shot_rank(image: Image.Image, labels: list[str], model, processor, device: str):
    prompts = [f"a photo of {lbl}" for lbl in labels]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # (1, num_text)
        probs = logits_per_image.softmax(dim=1).cpu().numpy().reshape(-1)
    return probs  # aligned with labels order

st.title("🍽️ Food Nutrition Detector")
st.caption("Predict dish from image → estimate nutrition per serving (approximate).")

colL, colR = st.columns([1.1, 1.2])

with colL:
    st.subheader("1) Upload food photo")
    up = st.file_uploader("PNG/JPG", type=["png","jpg","jpeg"])
    if up:
        image = Image.open(up).convert("RGB")
    else:
        image = Image.open(SAMPLE_IMG).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)

with colR:
    labels, df = load_labels_and_df()
    model, processor, device = load_clip()

    st.subheader("2) Predict dish")
    with st.spinner("Scoring with CLIP..."):
        probs = clip_zero_shot_rank(image, labels, model, processor, device)
    topk = min(5, len(labels))
    idxs = np.argsort(-probs)[:topk]
    top_items = [(labels[i], float(probs[i])) for i in idxs]

    st.write("**Top predictions:**")
    for name, p in top_items:
        st.write(f"- {name} — **{p*100:.1f}%**")

    st.divider()
    st.subheader("3) Confirm or search dish")
    default_choice = top_items[0][0] if top_items else ""
    selected_name = st.selectbox("Choose the dish (or type to search):", options=labels, index=labels.index(default_choice) if default_choice in labels else 0)
    manual = st.text_input("Or type a custom name (we'll try fuzzy match):", "")

    final_name = selected_name
    if manual.strip():
        final_name = manual.strip()
        matches = fuzzy_lookup(final_name, df, limit=5)
        best = matches[0] if matches else None
        if best:
            st.info(f"Closest match: **{best[0]}** (similarity {best[1]:.0f})")
            final_name = best[0]

    st.divider()
    st.subheader("4) Portion size")
    grams = st.slider("Portion (grams)", min_value=50, max_value=800, value=250, step=10)

    nutri = get_nutrition_for(final_name, df)
    if nutri is None:
        st.error("No nutrition record found. Please edit `data/nutrition_db.csv` to add this dish.")
    else:
        per_serving = scale_per_serving(nutri, grams)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Calories (kcal)", per_serving["calories_kcal"])
        c2.metric("Protein (g)", per_serving["protein_g"])
        c3.metric("Fat (g)", per_serving["fat_g"])
        c4.metric("Carbs (g)", per_serving["carbs_g"])

        st.caption(f"Base values per 100g — kcal: {nutri.calories}, protein: {nutri.protein} g, fat: {nutri.fat} g, carbs: {nutri.carbs} g")

    st.divider()
    st.subheader("5) Export & Actions")
    col_export, col_reset = st.columns(2)
    
    with col_export:
        if st.button("Download CSV (this prediction)", use_container_width=True):
            if nutri is None:
                st.warning("Nothing to export.")
            else:
                out_df = pd.DataFrame([{
                    "image_file": up.name if up else "sample.jpg",
                    "predicted_food": final_name,
                    "portion_g": grams,
                    **per_serving
                }])
                csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("Save CSV", data=csv, file_name="nutrition_estimate.csv", mime="text/csv")
    
    with col_reset:
        if st.button("🔄 Reset Form", use_container_width=True, type="secondary"):
            st.rerun()

st.sidebar.header("About")
st.sidebar.write("This demo uses zero-shot **CLIP** to recognize a dish among the names listed in your nutrition DB, then scales per‑100g values by your chosen portion size.")
st.sidebar.write("Extend `data/nutrition_db.csv` to improve coverage.")
