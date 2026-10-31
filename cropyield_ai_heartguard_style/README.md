# CropYield AI — Onion & Sugarcane

Local Flask application for crop-yield estimation, prediction history, analytics, and on-demand SHAP explanations.

## Run on Windows (VS Code / PowerShell)

1. Extract this ZIP. Open the **cropyield_ai_heartguard_style** folder in VS Code (the folder containing `app.py` and `requirements.txt`).
2. Confirm the dataset is in `data/pune_crop_yield_part_1.csv`. The app also checks common filename variants such as `pune_crop_yield_part_1(2).csv`.
3. Create a virtual environment using an installed Python version (Python 3.11 is recommended):

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python app.py
   ```

   If using Python 3.13, replace `py -3.11` with `py -3.13`; if a dependency fails to install, use Python 3.11.
4. Open `http://127.0.0.1:5000` in your browser. The first launch may take a few minutes because the two crop models are trained if model files are not present.

## Features

- Dashboard with dataset and model status
- Crop-specific Onion and Sugarcane prediction form
- Example form fill and reset controls
- On-demand SHAP explanation after prediction
- Local prediction history with delete and clear controls
- Dataset analytics and model metrics
- Responsive layout for smaller screens

## Dataset / model limitations

The app filters the dataset to Onion and Sugarcane. Check the `record_type` column: if rows are marked /, results are experimental and must not be treated as validated real-farm predictions. Sugarcane's current validation performance may be weak; do not use the output as agronomic advice without independent real-world validation. SHAP indicates how features influence this model's prediction, not causation.

Prediction history is stored locally in `cropyield_history.db`. Generated model files and metrics are stored in `models/`.
