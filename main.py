import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Models expected by Frontend
# ---------------------------------------------------------
class Transaction(BaseModel):
    trans_date_trans_time: str
    cc_num: str
    merchant: str
    category: str
    amt: float
    gender: str
    city: str
    state: str
    zip: str
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    unix_time: int
    merch_lat: float
    merch_long: float

class PredictRequest(BaseModel):
    transactions: List[Transaction]
    model: str = "ensemble"
    return_all: bool = False

# ---------------------------------------------------------
# Load ML Models if available
# ---------------------------------------------------------
models_loaded = []
lgb_model = None
xgb_model = None
cb_model = None
label_encoders = None
selected_features = None

try:
    if os.path.exists("lgb_final_model.pkl"):
        lgb_model = joblib.load("lgb_final_model.pkl")
        models_loaded.append("LightGBM")
    
    if os.path.exists("xgb_final_model.json"):
        import xgboost as xgb
        xgb_model = xgb.Booster()
        xgb_model.load_model("xgb_final_model.json")
        models_loaded.append("XGBoost")
        
    if os.path.exists("cb_final_model.cbm"):
        from catboost import CatBoostClassifier
        cb_model = CatBoostClassifier()
        cb_model.load_model("cb_final_model.cbm")
        models_loaded.append("CatBoost")
        
    if os.path.exists("label_encoders.pkl"):
        label_encoders = joblib.load("label_encoders.pkl")
        
    if os.path.exists("selected_features.json"):
        with open("selected_features.json", "r") as f:
            selected_features = json.load(f)
            
except Exception as e:
    print(f"Warning: Failed to load some ML models: {e}")

# In-memory history for rolling features across API calls
# Maps card_uid -> list of previous transaction dictionaries
card_history = {}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def preprocess_transaction(tx: Transaction) -> dict:
    global card_history
    
    dt = pd.to_datetime(tx.trans_date_trans_time)
    dob = pd.to_datetime(tx.dob)
    
    card_uid = f"{tx.cc_num}_{tx.dob}_{tx.zip}"
    merchant_clean = tx.merchant.replace("fraud_", "").strip()
    
    # Base extracted features
    feat = {
        "hour": dt.hour,
        "day_of_week": dt.dayofweek,
        "month": dt.month,
        "is_weekend": 1 if dt.dayofweek >= 5 else 0,
        "age": (dt - dob).days / 365.25,
        "distance_km": haversine_km(tx.lat, tx.long, tx.merch_lat, tx.merch_long),
        "log_amt": np.log1p(tx.amt),
        "log_city_pop": np.log1p(tx.city_pop),
        "is_night": 1 if (dt.hour <= 5 or dt.hour >= 23) else 0,
        "amt_x_hour": tx.amt * dt.hour,
        "amt_per_txday": tx.amt / (dt.dayofweek + 1)
    }

    # Historical state updates
    history = card_history.get(card_uid, [])
    if len(history) > 0:
        last_tx = history[-1]
        feat["uid_time_since_last"] = tx.unix_time - last_tx["unix_time"]
        feat["log_time_gap"] = np.log1p(feat["uid_time_since_last"])
        feat["uid_dist_from_prev"] = haversine_km(tx.lat, tx.long, last_tx["lat"], last_tx["long"])
        
        # Calculate recent activity counts
        v_24h, v_1h = 0, 0
        for p in reversed(history):
            if tx.unix_time - p["unix_time"] <= 86400: v_24h += 1
            if tx.unix_time - p["unix_time"] <= 3600: v_1h += 1
            if tx.unix_time - p["unix_time"] > 86400: break
            
        feat["uid_txn_24h"] = v_24h
        feat["uid_txn_7d"] = len(history) # Rough proxy for 7d if we don't store 7 days
        feat["txn_ratio"] = v_1h / (v_24h + 1)
        
        # Rolling amounts
        past_3_amts = [p["amt"] for p in history[-3:]]
        feat["amt_roll_mean_3"] = np.mean(past_3_amts)
        feat["amt_roll_std_3"] = np.std(past_3_amts) if len(past_3_amts) > 1 else 0
        feat["amt_vs_recent"] = tx.amt - feat["amt_roll_mean_3"]
        feat["amt_change"] = tx.amt / (feat["amt_roll_mean_3"] + 1e-6)
    else:
        # Default for first transaction
        feat["uid_time_since_last"] = 0
        feat["log_time_gap"] = 0
        feat["uid_dist_from_prev"] = 0
        feat["uid_txn_24h"] = 0
        feat["uid_txn_7d"] = 0
        feat["txn_ratio"] = 0
        feat["amt_roll_mean_3"] = tx.amt # Fallback
        feat["amt_roll_std_3"] = 0
        feat["amt_vs_recent"] = 0
        feat["amt_change"] = 1.0

    # Categorical Fallbacks
    feat["uid_unique_locs"] = 1
    feat["uid_loc_std_lat"] = 0.0
    feat["uid_loc_std_lon"] = 0.0
    feat["amt_zscore_global"] = 0.0 # Requires training set global mean/std
    feat["distance_anomaly"] = 1.0  # Requires training set distance mean
    
    # Store this transaction to history (limit to last 50 for memory safety)
    history.append({"unix_time": tx.unix_time, "lat": tx.lat, "long": tx.long, "amt": tx.amt})
    card_history[card_uid] = history[-50:]

    # Label Encoding
    cat_cols = {"merchant_clean": merchant_clean, "category": tx.category, "gender": tx.gender, "state": tx.state, "job": tx.job}
    if label_encoders:
        for k, val in cat_cols.items():
            if k in label_encoders:
                le = label_encoders[k]
                if val in le.classes_:
                    feat[k] = le.transform([val])[0]
                else:
                    feat[k] = le.transform(["<unseen>"])[0] if "<unseen>" in le.classes_ else -1

    return feat

@app.post("/predict")
async def predict_fraud(request: PredictRequest):
    predictions = []
    
    for tx in request.transactions:
        # If models missing, use a safe mock predictor so the UI still works
        if len(models_loaded) == 0:
            prob = 0.85 if tx.amt > 300 else 0.05
            predictions.append({
                "fraud_probability": prob,
                "is_fraud_predicted": prob > 0.5,
                "threshold": 0.5,
                "model_used": "mock_no_models_loaded"
            })
            continue
            
        feat_dict = preprocess_transaction(tx)
        
        # Prepare vector
        if selected_features:
            x_vec = np.array([[feat_dict.get(f, 0.0) for f in selected_features]], dtype=np.float32)
        else:
            x_vec = np.array([[feat_dict.get(k, 0.0) for k in feat_dict.keys()]], dtype=np.float32)

        # Predict based on requested model
        prob = 0.0
        used = request.model
        
        if request.model == "LightGBM" and lgb_model:
            prob = lgb_model.predict_proba(x_vec)[0][1]
        elif request.model == "XGBoost" and xgb_model:
            dmat = xgb.DMatrix(x_vec)
            prob = xgb_model.predict(dmat)[0]
        elif request.model == "CatBoost" and cb_model:
            prob = cb_model.predict_proba(x_vec)[0][1]
        else: # Ensemble
            p_list = []
            if lgb_model: p_list.append(lgb_model.predict_proba(x_vec)[0][1])
            if xgb_model: dmat = xgb.DMatrix(x_vec); p_list.append(xgb_model.predict(dmat)[0])
            if cb_model: p_list.append(cb_model.predict_proba(x_vec)[0][1])
            if len(p_list) > 0:
                prob = sum(p_list) / len(p_list)
                used = "ensemble"
                
        status = "Safe"
        is_fraud = False
        threshold = 0.97 if used == "ensemble" else 0.95
        
        # Tiered threshold logic
        if used == "ensemble":
            if prob >= 0.97:
                status = "Fraud"
                is_fraud = True
            elif prob >= 0.90:
                status = "Uncertain"
        else:
            # Custom thresholds for the individual models
            if prob >= 0.95:
                status = "Fraud"
                is_fraud = True
            elif prob >= 0.85:
                status = "Uncertain"

        predictions.append({
            "fraud_probability": float(prob),
            "is_fraud_predicted": is_fraud,
            "status": status,
            "threshold": threshold,
            "model_used": used
        })

    return {
        "predictions": predictions,
        "n_transactions": len(request.transactions)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "n_features": len(selected_features) if selected_features else 0, 
        "models_loaded": models_loaded if models_loaded else ["MOCK_MODE"],
        "card_registry_size": len(card_history)
    }

@app.get("/metadata")
async def get_metadata():
    return {
        "thresholds": {"ensemble": 0.5, "LightGBM": 0.5, "XGBoost": 0.5, "CatBoost": 0.5},
        "cat_feature_indices": []
    }

@app.post("/reset_card_history")
async def reset_history():
    global card_history
    card_history.clear()
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
