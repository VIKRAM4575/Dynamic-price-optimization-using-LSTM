import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# --- 1. CONFIGURATION & CONSTANTS ---
DATASET_PATH = r"C:\Users\vikra\Downloads\price\Dataset_limited.csv"
SAVE_DIR = "model_artifacts"
SEQ_LEN = 3 
EPOCHS = 100

# Reinforcement / Business Logic Constants
MIN_MARGIN = 0.05
DEMAND_SENSITIVITY = 0.15 
COMPETITOR_GAP = 0.005 

os.makedirs(SAVE_DIR, exist_ok=True)

class PricingSystem:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.model = None
        self.feature_cols = []

    # --- 2. DATA PREPARATION ---
    def prepare_and_train(self, df):
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Encoding categorical features
        for col in ['category', 'product_name', 'brand', 'season']:
            le = LabelEncoder()
            df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        self.feature_cols = [
            'base_price', 'cost_price', 'competitor_price', 
            'demand_factor', 'inventory_level', 'purchase_frequency',
            'category_enc', 'brand_enc', 'season_enc'
        ]
        
        # Scaling
        all_cols = self.feature_cols + ['dynamic_price']
        scaled_data = self.scaler.fit_transform(df[all_cols])
        
        # Create Sequences for LSTM
        X, y = [], []
        indices = df['product_name_enc'].values
        for prod in np.unique(indices):
            prod_data = scaled_data[indices == prod]
            if len(prod_data) > SEQ_LEN:
                for i in range(len(prod_data) - SEQ_LEN):
                    X.append(prod_data[i : i + SEQ_LEN, :-1])
                    y.append(prod_data[i + SEQ_LEN, -1])
        
        X, y = np.array(X), np.array(y)

        # --- 3. MODEL ARCHITECTURE ---
        self.model = Sequential([
            Input(shape=(X.shape[1], X.shape[2])),
            LSTM(32, return_sequences=False),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
        print("Starting Model Training...")
        self.model.fit(X, y, epochs=EPOCHS, batch_size=16, verbose=0)
        print("Training Complete.")
        
        # Save Artifacts
        self.model.save(os.path.join(SAVE_DIR, "pricing_model.keras"))
        with open(os.path.join(SAVE_DIR, "metadata.pkl"), 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'encoders': self.label_encoders,
                'features': self.feature_cols
            }, f)

    # --- 4. THE REINFORCE LAYER (Logic Layer) ---
    def apply_reinforcement_logic(self, ai_price, cost, comp, target_demand):
        """
        This function acts as the 'Reinforce Layer' by applying 
        economic constraints to the raw AI output.
        """
        # A. Demand Reinforcement
        demand_modifier = 1 + (target_demand - 0.5) * DEMAND_SENSITIVITY
        final_price = ai_price * demand_modifier

        # B. Competitor Constraint (Cap)
        final_price = min(final_price, comp * (1 - COMPETITOR_GAP))
        
        # C. Profit Margin Constraint (Floor)
        final_price = max(final_price, cost * (1 + MIN_MARGIN))
        
        return final_price

    def predict_dynamic_price(self, input_dict):
        # Prepare input for AI
        input_df = pd.DataFrame([input_dict])[self.feature_cols]
        input_df['dynamic_price'] = 0 # Placeholder for scaler
        
        scaled_row = self.scaler.transform(input_df)[0, :-1]
        pred_seq = np.tile(scaled_row, (1, SEQ_LEN, 1))
        
        # Raw AI Prediction
        ai_p_scaled = self.model.predict(pred_seq, verbose=0)[0, 0]
        
        # Denormalize
        d = np.zeros((1, len(self.feature_cols) + 1))
        d[0, -1] = ai_p_scaled
        ai_price_raw = self.scaler.inverse_transform(d)[0, -1]

        # Apply Reinforce Layer
        final_price = self.apply_reinforcement_logic(
            ai_price_raw, 
            input_dict['cost_price'], 
            input_dict['competitor_price'], 
            input_dict['demand_factor']
        )
        return final_price

# --- 5. EXECUTION ---
if __name__ == "__main__":
    system = PricingSystem()
    
    if os.path.exists(DATASET_PATH):
        raw_data = pd.read_csv(DATASET_PATH)
        system.prepare_and_train(raw_data)
        
        # Test Prediction with Reinforce Logic
        test_input = {
            'base_price': 1000,
            'cost_price': 800,
            'competitor_price': 950,
            'demand_factor': 0.8, # High Demand
            'inventory_level': 50,
            'purchase_frequency': 5,
            'category_enc': 0, 'brand_enc': 0, 'season_enc': 0
        }
        
        result = system.predict_dynamic_price(test_input)
        print(f"\nFinal Reinforcement-Adjusted Price: ₹{result:.2f}")
    else:
        print("Dataset not found. Check path.")