import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import json
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class MSMEScorer:
    def __init__(self, data_path="./mock_data"):
        self.data_path = data_path
        self.model = None
        self.explainer = None
        self.features_list = ['upi_velocity', 'gst_timeliness', 'pat_margin', 'asset_liability_ratio', 'sector_encoded']
        self.le = LabelEncoder()

    def engineer_features(self):
        # Load Data
        with open(f"{self.data_path}/business_master_profiles.json", "r") as f:
            profiles = json.load(f)
        upi_df = pd.read_csv(f"{self.data_path}/upi_logs.csv")
        gst_df = pd.read_csv(f"{self.data_path}/gst_filings.csv")

        # 1. UPI Velocity (Frequency of transactions)
        upi_counts = upi_df.groupby('sender_id').size().reset_index(name='upi_velocity')

        # 2. GST Timeliness (Inverted delay days)
        gst_perf = gst_df.groupby('business_id')['delay_days'].mean().reset_index()
        gst_perf['gst_timeliness'] = 10 - gst_perf['delay_days'] # Higher is better

        # Flatten Profiles
        rows = []
        for p in profiles:
            rows.append({
                'business_id': p['business_id'],
                'sector': p['industry_context']['sector'],
                'pat_margin': p['financials']['pat_margin'],
                'asset_liability_ratio': p['financials']['assets_to_liabilities']
            })
        
        base_df = pd.DataFrame(rows)
        base_df['sector_encoded'] = self.le.fit_transform(base_df['sector'])

        # Merge all features
        df = base_df.merge(upi_counts, left_on='business_id', right_on='sender_id', how='left')
        df = df.merge(gst_perf, left_on='business_id', right_on='business_id', how='left')
        
        # Fill NaNs for new businesses
        df.fillna({'upi_velocity': 0, 'gst_timeliness': 5}, inplace=True)
        return df

    def train_model(self):
        df = self.engineer_features()
        
        # Create a synthetic Target (300-900) based on features for the hackathon
        # In a real scenario, this would be historical default data
        y = (df['pat_margin'] * 20) + (df['gst_timeliness'] * 40) + (df['upi_velocity'] * 2) + 300
        y = y.clip(300, 900)

        X = df[self.features_list]
        
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        self.model.fit(X, y)
        
        # Initialize SHAP
        self.explainer = shap.TreeExplainer(self.model)
        self.model.save_model("msme_model.json")
        self.processed_df = df # Keep for lookup
        print("Model trained and SHAP explainer initialized.")

    def get_score_and_shap(self, gstin):
        """Returns (Score, SHAP_Dict) for a specific GSTIN."""
        row = self.processed_df[self.processed_df['business_id'] == gstin]
        if row.empty:
            return None, None

        X_input = row[self.features_list]
        score = float(self.model.predict(X_input)[0])
        
        shap_values = self.explainer.shap_values(X_input)
        shap_dict = dict(zip(self.features_list, shap_values[0].tolist()))
        
        return round(score, 2), shap_dict

if __name__ == "__main__":
    scorer = MSMEScorer()
    scorer.train_model()
    
    # Test with the first business from mock data
    sample_gstin = scorer.processed_df['business_id'].iloc[0]
    score, shaps = scorer.get_score_and_shap(sample_gstin)
    
    print(f"\nResults for {sample_gstin}:")
    print(f"Credit Score: {score}")
    print(f"SHAP Contributions: {json.dumps(shaps, indent=2)}")