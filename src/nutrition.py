from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from rapidfuzz import process, fuzz

@dataclass
class NutritionPer100g:
    calories: float
    protein: float
    fat: float
    carbs: float

def load_nutrition_db(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # basic cleanup
    df["food_name"] = df["food_name"].str.strip()
    return df

def fuzzy_lookup(food_name: str, df: pd.DataFrame, limit: int = 5) -> List[Tuple[str, float]]:
    choices = df["food_name"].tolist()
    matches = process.extract(food_name, choices, scorer=fuzz.WRatio, limit=limit)
    # returns list of (match_name, score, idx); we reduce to (name, score)
    return [(m[0], float(m[1])) for m in matches]

def get_nutrition_for(food_name: str, df: pd.DataFrame) -> NutritionPer100g | None:
    row = df.loc[df["food_name"].str.lower() == food_name.lower()]
    if row.empty:
        return None
    r = row.iloc[0]
    return NutritionPer100g(
        calories=float(r["calories_kcal_100g"]),
        protein=float(r["protein_g_100g"]),
        fat=float(r["fat_g_100g"]),
        carbs=float(r["carbs_g_100g"]),
    )

def scale_per_serving(nutri_100g: NutritionPer100g, grams: float) -> Dict[str, float]:
    factor = max(grams, 0.0) / 100.0
    return {
        "calories_kcal": round(nutri_100g.calories * factor, 2),
        "protein_g": round(nutri_100g.protein * factor, 2),
        "fat_g": round(nutri_100g.fat * factor, 2),
        "carbs_g": round(nutri_100g.carbs * factor, 2),
    }
