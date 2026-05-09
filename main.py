import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

if not os.path.exists("images"):
    os.makedirs("images")

def main():
    print("🚀 Starting Fraud Detection Pipeline...\n")

    # 1. Load the Data
    df = pd.read_csv("credit_card_data.csv")
    
    # Show the imbalance
    fraud_count = df['Is_Fraud'].sum()
    print(f"📊 Dataset contains {len(df)} transactions. Only {fraud_count} are fraud.")

    # 2. Prepare Features (X) and Target (y)
    X = df.drop(columns=['Is_Fraud'])
    y = df['Is_Fraud']

    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train the Model
    # Notice we are using RandomForestCLASSIFIER now, not Regressor!
    print("\n🧠 Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Make Predictions
    predictions = model.predict(X_test)

    # 5. Evaluate the Results
    print("\n📝 Classification Report:")
    # The classification report shows Precision, Recall, and F1-Score
    print(classification_report(y_test, predictions, target_names=['Legit (0)', 'Fraud (1)']))

    # 6. Generate a Confusion Matrix
    print("🎨 Generating Confusion Matrix Visual...")
    cm = confusion_matrix(y_test, predictions)
    
    plt.figure(figsize=(8, 6))
    # We use a heatmap to make the matrix look professional
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Predicted Legit', 'Predicted Fraud'],
                yticklabels=['Actual Legit', 'Actual Fraud'])
    
    plt.title("Fraud Detection Confusion Matrix")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("images/confusion_matrix.png")
    print("✅ Matrix saved to images/confusion_matrix.png")
    print("\n🎉 Pipeline Complete!")

if __name__ == "__main__":
    main()