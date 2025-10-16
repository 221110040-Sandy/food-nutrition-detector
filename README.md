# Food Nutrition Detector (Image → Dish → Nutrition)

A Streamlit app that predicts the **dish** from an image using **zero-shot CLIP**
and estimates its **nutrition per serving** using a curated nutrition table.

> Works fully on CPU. First run will download CLIP weights (~400MB).

## 🚀 Quickstart

```bash
# 1) (Optional but recommended) create virtual env
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

Then open the local URL shown in your terminal.

## 🧠 How it works

- We use **CLIP (openai/clip-vit-base-patch32)** to score the uploaded image against a **fixed label set** derived from `data/nutrition_db.csv` (the dish names become the class prompts).
- The app shows **Top‑5 predictions** with probabilities. You can **confirm/override** the label.
- Nutrition values are looked up **per 100 g** and scaled by your **portion size** (you can use the slider).

## 📦 Project Structure

```
food-nutrition-detector/
├── app.py
├── src/
│   └── nutrition.py
├── data/
│   └── nutrition_db.csv        # extend/edit with your own dishes
├── assets/
│   └── sample.jpg              # example image
├── requirements.txt
└── README.md
```

## 📝 Extending the database

Edit `data/nutrition_db.csv` and add new rows. Columns:

- `food_name` (string, unique)
- `calories_kcal_100g`, `protein_g_100g`, `fat_g_100g`, `carbs_g_100g` (floats)
- `category` (optional grouping)

> The values are **approximate** per 100 grams; adjust for your locale or lab data.

## ⚠️ Disclaimer

Predictions and nutrition numbers are **estimates** and may be inaccurate.
Use at your own risk. Not a medical device.
