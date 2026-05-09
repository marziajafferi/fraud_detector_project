import pandas as pd
import numpy as np

def create_fraud_dataset(filename="credit_card_data.csv", num_transactions=2000):
    np.random.seed(42)
    print("⚙️ Generating MESSY, real-world banking data...")
    
    # Generate legitimate transactions (95%)
    num_legit = int(num_transactions * 0.95)
    # MESSY: Some legit people buy very expensive things (up to $5000)
    legit_amount = np.random.exponential(scale=100, size=num_legit).clip(1, 5000)
    legit_distance = np.random.exponential(scale=20, size=num_legit)
    legit_pin_matched = np.random.choice([1, 0], size=num_legit, p=[0.90, 0.10])
    legit_online = np.random.choice([1, 0], size=num_legit, p=[0.5, 0.5])
    
    # Generate fraudulent transactions (5%)
    num_fraud = num_transactions - num_legit
    # MESSY: Fraudsters are now testing small amounts to avoid detection
    fraud_amount = np.random.normal(loc=250, scale=200, size=num_fraud).clip(10, 5000)
    fraud_distance = np.random.normal(loc=50, scale=50, size=num_fraud).clip(1, 1000)
    fraud_pin_matched = np.random.choice([1, 0], size=num_fraud, p=[0.3, 0.7])
    fraud_online = np.random.choice([1, 0], size=num_fraud, p=[0.8, 0.2])
    
    # Combine the data
    df_legit = pd.DataFrame({
        'Transaction_Amount': legit_amount,
        'Distance_From_Home': legit_distance,
        'Pin_Matched': legit_pin_matched,
        'Online_Transaction': legit_online,
        'Is_Fraud': 0
    })
    
    df_fraud = pd.DataFrame({
        'Transaction_Amount': fraud_amount,
        'Distance_From_Home': fraud_distance,
        'Pin_Matched': fraud_pin_matched,
        'Online_Transaction': fraud_online,
        'Is_Fraud': 1
    })
    
    df = pd.concat([df_legit, df_fraud]).sample(frac=1).reset_index(drop=True)
    
    df['Transaction_Amount'] = np.round(df['Transaction_Amount'], 2)
    df['Distance_From_Home'] = np.round(df['Distance_From_Home'], 1)
    
    df.to_csv(filename, index=False)
    print(f"✅ Generated {filename} with {num_legit} legit and {num_fraud} fraud records.")

if __name__ == "__main__":
    create_fraud_dataset()