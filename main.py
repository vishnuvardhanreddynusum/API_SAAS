from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import os
from typing import List, Dict, Any
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="NexaCloud Churn Prediction API",
    description="API for predicting SaaS customer churn using XGBoost model",
    version="1.0.0"
)

# Load model artefacts at startup
MODEL_PATH = "model_artefacts/xgb_churn_model.pkl"
SCALER_PATH = "model_artefacts/scaler.pkl"
FEATURES_PATH = "model_artefacts/selected_features.pkl"
THRESHOLD_PATH = "model_artefacts/threshold.pkl"

# Global variables to store loaded artefacts
model = None
scaler = None
selected_features = None
threshold = None

def load_artefacts():
    """Load all required model artefacts"""
    global model, scaler, selected_features, threshold
    
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        selected_features = joblib.load(FEATURES_PATH)
        threshold = joblib.load(THRESHOLD_PATH)
        print("✅ All model artefacts loaded successfully")
    except Exception as e:
        print(f"❌ Error loading artefacts: {e}")
        raise

# Load artefacts when the app starts
@app.on_event("startup")
async def startup_event():
    load_artefacts()

# Define request body structure
class ChurnPredictionRequest(BaseModel):
    """Input features for churn prediction"""
    # Customer information
    enterprise_tier: str  # 'SMB' or 'Enterprise'
    has_crm_integration: str  # 'Yes' or 'No'
    has_sub_accounts: str  # 'Yes' or 'No'
    subscription_months: int
    voice_calling_addon: str  # 'Yes' or 'No'
    multi_workspace_addon: str  # 'Yes', 'No', or 'Not Applicable'
    plan_type: str  # 'Starter', 'Pro', or 'Free'
    sso_enabled: str  # 'Yes' or 'No'
    auto_backup_enabled: str  # 'Yes' or 'No'
    endpoint_security_enabled: str  # 'Yes' or 'No'
    priority_support_enabled: str  # 'Yes' or 'No'
    live_collab_enabled: str  # 'Yes' or 'No'
    media_vault_enabled: str  # 'Yes' or 'No'
    billing_cycle: str  # 'Monthly', 'Annual', or 'Biennial'
    e_invoicing_enabled: str  # 'Yes' or 'No'
    payment_method: str  # 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    mrr_usd: float
    ltv_usd: float

class PredictionResponse(BaseModel):
    """Prediction response structure"""
    churn_probability: float
    churn_prediction: int  # 1 = will churn, 0 = will retain
    churn_risk: str  # 'High', 'Medium', or 'Low'
    threshold_used: float
    features_used: List[str]

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    customers: List[ChurnPredictionRequest]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to raw input data"""
    
    # Fill NaN values in subscription_months
    df['subscription_months'] = df['subscription_months'].fillna(0)
    
    # Tenure / Lifecycle Features
    df['subscription_group'] = pd.cut(
        df['subscription_months'],
        bins=[0, 12, 24, 48, float('inf')],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)
    
    df['is_first_year'] = (df['subscription_months'] <= 12).astype(int)
    df['is_very_new'] = (df['subscription_months'] <= 3).astype(int)
    df['is_long_term'] = (df['subscription_months'] >= 24).astype(int)
    df['is_contract_end'] = (df['subscription_months'] % 12 == 0).astype(int)
    
    # Monetary / Revenue Features
    df['avg_monthly_revenue'] = df.apply(
        lambda x: x['ltv_usd'] / x['subscription_months'] if x['subscription_months'] > 0
                  else x['mrr_usd'], axis=1
    )
    df['mrr_to_ltv_ratio'] = df['mrr_usd'] / (df['ltv_usd'] + 1)
    df['charge_increase_flag'] = (df['mrr_usd'] > df['avg_monthly_revenue']).astype(int)
    df['ltv_projection'] = df['ltv_usd'] + (df['mrr_usd'] * 6)
    df['price_to_tenure_ratio'] = df['mrr_usd'] / (df['subscription_months'] + 1)
    df['billing_efficiency'] = df['ltv_usd'] / (df['mrr_usd'] * df['subscription_months'] + 1)
    df['mrr_tier'] = pd.cut(df['mrr_usd'],
                            bins=[0, 35, 70, 105, float('inf')],
                            labels=[0, 1, 2, 3]).astype(int)
    
    # Service Engagement Features
    addon_cols = ['voice_calling_addon', 'sso_enabled', 'auto_backup_enabled',
                  'endpoint_security_enabled', 'priority_support_enabled',
                  'live_collab_enabled', 'media_vault_enabled']
    
    # Convert Yes/No columns
    for col in addon_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    df['num_addons'] = df[addon_cols].sum(axis=1)
    df['addon_adoption_rate'] = df['num_addons'] / (df['subscription_months'] + 1)
    
    df['has_security_bundle'] = (
        (df['sso_enabled'] == 1) & (df['endpoint_security_enabled'] == 1)
    ).astype(int)
    
    df['has_collab_bundle'] = (
        (df['live_collab_enabled'] == 1) & (df['media_vault_enabled'] == 1)
    ).astype(int)
    
    df['has_backup_support'] = (
        (df['auto_backup_enabled'] == 1) & (df['priority_support_enabled'] == 1)
    ).astype(int)
    
    df['is_pro_plan'] = (df['plan_type'] == 'Pro').astype(int)
    df['is_free_plan'] = (df['plan_type'] == 'Free').astype(int)
    df['is_starter_plan'] = (df['plan_type'] == 'Starter').astype(int)
    
    # Handle multi_workspace_addon
    df['multi_workspace_addon'] = df['multi_workspace_addon'].map(
        {'Yes': 1, 'No': 0, 'Not Applicable': 2}
    )
    
    # Contract & Payment Risk Features
    df['billing_risk'] = df['billing_cycle'].map({'Biennial': 0, 'Annual': 1, 'Monthly': 2})
    df['is_monthly_billing'] = (df['billing_cycle'] == 'Monthly').astype(int)
    
    df['payment_risk_score'] = df['payment_method'].map({
        'Electronic check': 3,
        'Mailed check': 2,
        'Bank transfer (automatic)': 1,
        'Credit card (automatic)': 1
    })
    
    df['is_electronic_check'] = (df['payment_method'] == 'Electronic check').astype(int)
    df['paperless_electronic_risk'] = (
        (df['e_invoicing_enabled'] == 'Yes') & (df['payment_method'] == 'Electronic check')
    ).astype(int)
    df['is_enterprise'] = (df['enterprise_tier'] == 'Enterprise').astype(int)
    
    # Interaction Features
    median_mrr = df['mrr_usd'].median()
    df['high_cost_new_account'] = (
        (df['mrr_usd'] > median_mrr) & (df['subscription_months'] < 12)
    ).astype(int)
    
    df['pro_monthly_risk'] = (
        (df['is_pro_plan'] == 1) & (df['is_monthly_billing'] == 1)
    ).astype(int)
    
    df['new_account_elec_check'] = (
        (df['subscription_months'] < 6) & (df['is_electronic_check'] == 1)
    ).astype(int)
    
    df['monthly_no_addons'] = (
        (df['is_monthly_billing'] == 1) & (df['num_addons'] == 0)
    ).astype(int)
    
    df['longterm_no_security'] = (
        (df['is_long_term'] == 1) & (df['has_security_bundle'] == 0)
    ).astype(int)
    
    df['many_addons_very_new'] = (
        (df['num_addons'] > 3) & (df['is_very_new'] == 1)
    ).astype(int)
    
    # Composite Scores
    df['engagement_score'] = (
        df['num_addons'] * 0.35 +
        (2 - df['billing_risk']) * 0.40 +
        (df['subscription_months'] / df['subscription_months'].max()) * 0.25
    )
    
    df['churn_risk_score'] = (
        df['is_monthly_billing'] * 0.30 +
        df['is_electronic_check'] * 0.20 +
        df['is_first_year'] * 0.20 +
        (df['mrr_usd'] > 70).astype(int) * 0.15 +
        (df['num_addons'] == 0).astype(int) * 0.15
    )
    
    # Encode binary Yes/No columns
    yes_no_cols = [
        'has_crm_integration', 'has_sub_accounts',
        'voice_calling_addon', 'sso_enabled', 'auto_backup_enabled',
        'endpoint_security_enabled', 'priority_support_enabled',
        'live_collab_enabled', 'media_vault_enabled', 'e_invoicing_enabled'
    ]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # One-hot encode remaining categoricals
    df = pd.get_dummies(df, drop_first=True)
    
    return df

def preprocess_input(data: ChurnPredictionRequest) -> np.ndarray:
    """Convert input data to features ready for prediction"""
    
    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Apply feature engineering
    df = engineer_features(df)
    
    # Ensure all selected features are present
    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select only the features used in training
    df = df[selected_features]
    
    # Scale features
    scaled_features = scaler.transform(df)
    
    return scaled_features

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "NexaCloud Churn Prediction API",
        "version": "1.0.0",
        "description": "Predict SaaS customer churn using XGBoost model",
        "endpoints": {
            "/": "This information",
            "/health": "Health check",
            "/predict": "Single prediction endpoint",
            "/predict/batch": "Batch prediction endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model artefacts not loaded")
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "features_loaded": selected_features is not None,
        "threshold": threshold
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ChurnPredictionRequest):
    """Predict churn for a single customer"""
    try:
        # Preprocess input
        features = preprocess_input(request)
        
        # Get probability
        churn_probability = model.predict_proba(features)[0][1]
        
        # Apply threshold for final prediction
        churn_prediction = 1 if churn_probability >= threshold else 0
        
        # Determine risk level
        if churn_probability >= 0.7:
            risk_level = "High"
        elif churn_probability >= 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return PredictionResponse(
            churn_probability=round(float(churn_probability), 4),
            churn_prediction=churn_prediction,
            churn_risk=risk_level,
            threshold_used=threshold,
            features_used=selected_features
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict churn for multiple customers"""
    try:
        predictions = []
        
        for customer in request.customers:
            # Preprocess input
            features = preprocess_input(customer)
            
            # Get probability
            churn_probability = model.predict_proba(features)[0][1]
            
            # Apply threshold
            churn_prediction = 1 if churn_probability >= threshold else 0
            
            # Determine risk level
            if churn_probability >= 0.7:
                risk_level = "High"
            elif churn_probability >= 0.3:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            predictions.append(PredictionResponse(
                churn_probability=round(float(churn_probability), 4),
                churn_prediction=churn_prediction,
                churn_risk=risk_level,
                threshold_used=threshold,
                features_used=selected_features
            ))
        
        return BatchPredictionResponse(predictions=predictions)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)