import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import json
import pandas as pd
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# Load dataset once at startup
df = pd.read_csv("penguins_base.csv")
df = df.dropna(subset=["species"])
df["sex"] = df["sex"].fillna("unknown")

# Precompute species stats for card display
SPECIES_INFO = {
    "Adelie": {
        "color": "#FF6B6B",
        "emoji": "🐧",
        "desc": "A espécie mais comum, encontrada nas três ilhas. Bico curto e robusto.",
        "island": ["Biscoe", "Dream", "Torgersen"],
    },
    "Chinstrap": {
        "color": "#4ECDC4",
        "emoji": "🐧",
        "desc": "Reconhecida pela faixa escura sob o bico, como uma cinta. Encontrada apenas em Dream.",
        "island": ["Dream"],
    },
    "Gentoo": {
        "color": "#45B7D1",
        "emoji": "🐧",
        "desc": "A maior das três espécies, com nadadeiras longas. Habita apenas Biscoe.",
        "island": ["Biscoe"],
    },
}


@app.route("/")
def index():
    # Compute ranges for slider config
    ranges = {}
    for col in ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]:
        ranges[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "step": 0.1 if col != "body_mass_g" and col != "flipper_length_mm" else (1.0 if col == "flipper_length_mm" else 25.0),
        }
    return render_template("index.html", ranges=ranges, species_info=SPECIES_INFO)


@app.route("/api/filter")
def filter_data():
    filtered = df.copy()

    island = request.args.get("island")
    if island and island != "all":
        filtered = filtered[filtered["island"] == island]

    sex = request.args.get("sex")
    if sex and sex != "all":
        filtered = filtered[filtered["sex"] == sex]

    bill_length = request.args.get("bill_length_mm")
    if bill_length:
        filtered = filtered[
            (filtered["bill_length_mm"] >= float(request.args.get("bill_length_min", df["bill_length_mm"].min()))) &
            (filtered["bill_length_mm"] <= float(request.args.get("bill_length_max", df["bill_length_mm"].max())))
        ]

    bill_depth = request.args.get("bill_depth_mm")
    if bill_depth:
        filtered = filtered[
            (filtered["bill_depth_mm"] >= float(request.args.get("bill_depth_min", df["bill_depth_mm"].min()))) &
            (filtered["bill_depth_mm"] <= float(request.args.get("bill_depth_max", df["bill_depth_mm"].max())))
        ]

    flipper = request.args.get("flipper_length_mm")
    if flipper:
        filtered = filtered[
            (filtered["flipper_length_mm"] >= float(request.args.get("flipper_min", df["flipper_length_mm"].min()))) &
            (filtered["flipper_length_mm"] <= float(request.args.get("flipper_max", df["flipper_length_mm"].max())))
        ]

    mass = request.args.get("body_mass_g")
    if mass:
        filtered = filtered[
            (filtered["body_mass_g"] >= float(request.args.get("mass_min", df["body_mass_g"].min()))) &
            (filtered["body_mass_g"] <= float(request.args.get("mass_max", df["body_mass_g"].max())))
        ]

    total = len(filtered)
    species_counts = filtered["species"].value_counts().to_dict()
    species_pct = {k: round(v / total * 100, 1) if total > 0 else 0 for k, v in species_counts.items()}

    # Stats for matched records
    stats = {}
    if total > 0:
        for sp in filtered["species"].unique():
            sp_df = filtered[filtered["species"] == sp]
            stats[sp] = {
                "count": int(species_counts.get(sp, 0)),
                "pct": species_pct.get(sp, 0),
                "bill_length": {"min": round(float(sp_df["bill_length_mm"].min()), 1), "max": round(float(sp_df["bill_length_mm"].max()), 1), "mean": round(float(sp_df["bill_length_mm"].mean()), 1)},
                "bill_depth": {"min": round(float(sp_df["bill_depth_mm"].min()), 1), "max": round(float(sp_df["bill_depth_mm"].max()), 1), "mean": round(float(sp_df["bill_depth_mm"].mean()), 1)},
                "flipper": {"min": round(float(sp_df["flipper_length_mm"].min()), 1), "max": round(float(sp_df["flipper_length_mm"].max()), 1), "mean": round(float(sp_df["flipper_length_mm"].mean()), 1)},
                "mass": {"min": round(float(sp_df["body_mass_g"].min()), 0), "max": round(float(sp_df["body_mass_g"].max()), 0), "mean": round(float(sp_df["body_mass_g"].mean()), 0)},
                "islands": sp_df["island"].value_counts().to_dict(),
                "sex": sp_df["sex"].value_counts().to_dict(),
            }

    return jsonify({
        "total": total,
        "species_counts": species_counts,
        "species_pct": species_pct,
        "stats": stats,
    })


if __name__ == "__main__":
    print("=" * 50)
    print("  🐧 Penguin Explorer rodando em:")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
