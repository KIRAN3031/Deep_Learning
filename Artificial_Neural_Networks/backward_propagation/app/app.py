import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# --- LOAD DATA AND PREPROCESSORS ---
@st.cache_resource
def load_resources():
    # 1. Load the original data to replicate the exact training columns
    if not os.path.exists('CarPrice_dataset.csv'):
        st.error("Dataset 'CarPrice_dataset' not found!")
        return None, None, None, None, None
        
    df = pd.read_csv('CarPrice_dataset.csv')
    df.drop_duplicates(inplace=True)
    
    # Identify categorical columns (objects in the CSV)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'price' in cat_cols: cat_cols.remove('price')
    
    # 2. Replicate Training Preprocessing to get column structure
    df_encoded = df.copy()
    for col in cat_cols:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
    
    # Note: Training used get_dummies on a copy, but then standardized the category codes
    # We will follow the standardization of the encoded columns as per Task 3
    scaler = StandardScaler()
    numerical_cols = df_encoded.columns # Includes 'price'
    scaler.fit(df_encoded[numerical_cols])
    
    # 3. Load the trained Keras model
    try:
        # Looking for the .h5 format common in Keras
        model = tf.keras.models.load_model('trained_model.keras')
    except Exception as e:
        st.error(f"Model file error: {e}")
        model = None
        
    return df, scaler, model, cat_cols, df_encoded.columns.tolist()

# Initialize resources
df_raw, scaler, model, cat_cols, all_encoded_cols = load_resources()

# --- CUSTOM CSS ---
if os.path.exists("styles.css"):
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- UI HEADER ---
st.title("🚗 AI Car Price Predictor")
st.markdown("Provide vehicle details to get a Deep Learning powered price estimate.")
st.write("---")

if model is not None:
    # --- USER INPUTS ---
    col1, col2 = st.columns(2)
    user_inputs = {}

    with col1:
        st.subheader("General Info")
        for col in cat_cols:
            options = sorted(df_raw[col].unique().tolist())
            user_inputs[col] = st.selectbox(f"{col.title()}", options)

    with col2:
        st.subheader("Technical Specs")
        # Numerical features are those in df_encoded that aren't 'price' and weren't objects
        num_feats = [c for c in df_raw.columns if c != 'price' and c not in cat_cols]
        for col in num_feats:
            val_min = float(df_raw[col].min())
            val_max = float(df_raw[col].max())
            val_default = float(df_raw[col].mean())
            user_inputs[col] = st.number_input(f"{col.replace('_', ' ').title()}", 
                                              min_value=0.0, max_value=val_max*10, value=val_default)

    # --- PREDICTION LOGIC ---
    if st.button("Calculate Predicted Price"):
        try:
            # 1. Create a row matching the encoded dataframe structure (excluding price)
            input_row = {}
            for col in all_encoded_cols:
                if col == 'price':
                    input_row[col] = 0 # Placeholder for scaling
                elif col in cat_cols:
                    # Get the code for the selected category string
                    category_mapping = dict(enumerate(df_raw[col].astype('category').cat.categories))
                    # Reverse mapping to get code from string
                    inv_map = {v: k for k, v in category_mapping.items()}
                    input_row[col] = inv_map[user_inputs[col]]
                else:
                    input_row[col] = user_inputs[col]

            # Convert to DataFrame to ensure correct column order
            input_df = pd.DataFrame([input_row])[all_encoded_cols]
            
            # 2. Scale the entire row (matching the training scaler fit)
            scaled_data = scaler.transform(input_df)
            
            # 3. Extract Features for the Model (X)
            price_idx = all_encoded_cols.index('price')
            X_input = np.delete(scaled_data, price_idx, axis=1)
            
            # 4. Predict
            prediction_scaled = model.predict(X_input, verbose=0)
            
            # 5. Inverse Scale to get the Price
            # Create a dummy array for the scaler with the predicted value in the price column
            inverse_dummy = scaled_data.copy()
            inverse_dummy[0, price_idx] = prediction_scaled[0, 0]
            
            final_result_array = scaler.inverse_transform(inverse_dummy)
            predicted_price = final_result_array[0, price_idx]

            # 6. Display Result
            st.markdown(f"""
                <div class="prediction-container">
                    <p class="prediction-label">Estimated Market Value</p>
                    <h2 class="prediction-value">${max(0, predicted_price):,.2f}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            if predicted_price > 0:
                st.balloons()
            else:
                st.warning("The model predicts a negligible value for this configuration.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Check if the input values are within reasonable ranges.")

else:
    st.error("Application failed to load the model. Please check file paths.")

st.write("---")
st.caption("FNN Model Architecture: Input(64) → ReLU → Hidden(32) → ReLU → Output(1) Linear")