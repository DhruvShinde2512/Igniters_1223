import pandas as pd
import numpy as np
import json
import os
import random
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker and local directory
fake = Faker('en_IN') # Using Indian locale for GSTIN/Context
output_dir = "./mock_data"
os.makedirs(output_dir, exist_ok=True)

def generate_gstin():
    """Generates a mock GSTIN-like string."""
    return f"{random.randint(10, 37)}{fake.bothify(text='??PRP####?')}{random.randint(1, 9)}Z{random.randint(1, 9)}"

def create_msme_data(num_businesses=10):
    gstins = [generate_gstin() for _ in range(num_businesses)]
    businesses = []
    
    # 1. Generate Business Master Profiles (JSON)
    for gstin in gstins:
        profile = {
            "business_id": gstin,
            "industry_context": {
                "sector": random.choice(["Manufacturing", "Retail", "Services", "Textiles"]),
                "buyer_market_type": random.choice(["B2B", "B2C", "Export"]),
                "top_clients_share": round(random.uniform(10.0, 70.0), 2),
                "supply_chain_dependency": random.choice(["Low", "Medium", "High"])
            },
            "financials": {
                "monthly_gst": round(random.uniform(50000, 500000), 2),
                "pat_margin": round(random.uniform(2.0, 15.0), 2),
                "assets_to_liabilities": round(random.uniform(1.1, 3.5), 2)
            },
            "owner_profile": {
                "pan_status": "Verified",
                "failed_businesses": random.randint(0, 2),
                "dependents": random.randint(1, 5)
            }
        }
        businesses.append(profile)

    with open(f"{output_dir}/business_master_profiles.json", "w") as f:
        json.dump(businesses, f, indent=4)

    # 2. Generate UPI Logs (CSV) with Fraud Injection
    upi_records = []
    start_date = datetime.now() - timedelta(days=30)

    # Standard Random Transactions
    for _ in range(200):
        upi_records.append({
            "timestamp": fake.date_time_between(start_date=start_date),
            "sender_id": random.choice(gstins),
            "receiver_id": fake.uuid4()[:8], # External vendor
            "amount": round(random.uniform(500, 50000), 2)
        })

    # --- CIRCULAR FRAUD INJECTION (A -> B -> C -> A) ---
    # Pick 3 GSTINs from our pool (or create 2 dummy partners for the target)
    target_a = gstins[0]
    partner_b = "GSTIN_FRAUD_B"
    partner_c = "GSTIN_FRAUD_C"
    base_time = datetime.now() - timedelta(hours=12)
    
    fraud_loop = [
        {"timestamp": base_time, "sender_id": target_a, "receiver_id": partner_b, "amount": 100000.0},
        {"timestamp": base_time + timedelta(hours=1), "sender_id": partner_b, "receiver_id": partner_c, "amount": 99950.0},
        {"timestamp": base_time + timedelta(hours=2), "sender_id": partner_c, "receiver_id": target_a, "amount": 99900.0}
    ]
    upi_records.extend(fraud_loop)
    
    upi_df = pd.DataFrame(upi_records)
    upi_df.to_csv(f"{output_dir}/upi_logs.csv", index=False)

    # 3. Generate GST Filings (CSV)
    gst_records = []
    for gstin in gstins:
        for i in range(6): # 6 months of filings
            filing_date = (datetime.now() - timedelta(days=30*i)).replace(day=random.randint(10, 20))
            gst_records.append({
                "business_id": gstin,
                "filing_month": filing_date.strftime("%B %Y"),
                "filing_date": filing_date.strftime("%Y-%m-%d"),
                "delay_days": random.randint(0, 5) if random.random() > 0.8 else 0
            })
    
    gst_df = pd.DataFrame(gst_records)
    gst_df.to_csv(f"{output_dir}/gst_filings.csv", index=False)

    print(f"Successfully generated mock data for {num_businesses} MSMEs in {output_dir}")

if __name__ == "__main__":
    create_msme_data(10)