# src/api/services/ml_service.py

import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Dict, Any

# --- Configuração de Caminhos para o Modelo Salvo ---
# Estes caminhos devem ser relativos à RAIZ DO PROJETO,
# presumindo que o uvicorn será executado da raiz do projeto.
MODEL_PATH = 'models/ddos_classifier_rf_model.joblib'
SCALER_PATH = 'models/ddos_classifier_scaler.joblib'
LABEL_ENCODER_PATH = 'models/ddos_classifier_label_encoder.joblib'

# Uma lista com a ordem EXATA das features que o modelo de ML espera.
# Esta lista DEVE ser gerada e salva no seu script de treinamento (trainer.py)
# e carregada aqui, ou definida manualmente se a ordem for garantida.
# EXTREMAMENTE IMPORTANTE: A ORDEM DAS COLUNAS DEVE SER A MESMA DO TREINAMENTO.
# Por enquanto, placeholder com algumas. VOCÊ PRECISA COMPLETAR ISTO.
EXPECTED_FEATURES_ORDER = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
    'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
    'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s',
    'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count',
    'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length', # Fwd Header Length duplicated in your list, check your dataset
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
    'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std',
    'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]


# Variáveis globais para armazenar o modelo, scaler e label_encoder
# Serão carregadas uma vez quando o módulo for importado
ml_model = None
ml_scaler = None
ml_label_encoder = None

def load_ml_components():
    """
    Loads the trained ML model, scaler, and label encoder from disk.
    This function should be called once at API startup.
    """
    global ml_model, ml_scaler, ml_label_encoder
    
    if ml_model is not None and ml_scaler is not None and ml_label_encoder is not None:
        print("ML components already loaded.")
        return # Already loaded, prevent redundant loading

    print("Loading ML model, scaler, and label encoder...")
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {os.path.abspath(MODEL_PATH)}")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found: {os.path.abspath(SCALER_PATH)}")
        if not os.path.exists(LABEL_ENCODER_PATH):
            raise FileNotFoundError(f"Label Encoder file not found: {os.path.abspath(LABEL_ENCODER_PATH)}")

        ml_model = joblib.load(MODEL_PATH)
        ml_scaler = joblib.load(SCALER_PATH)
        ml_label_encoder = joblib.load(LABEL_ENCODER_PATH)
        print("ML components loaded successfully!")
        print(f"Loaded Label Encoder classes: {ml_label_encoder.classes_}")
    except Exception as e:
        print(f"Error loading ML components: {e}")
        raise # Rethrow to prevent API from starting without models

def predict_ddos_attack(features_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Receives raw features (from API request), preprocesses them, and makes a prediction.

    Args:
        features_data (Dict[str, Any]): A dictionary of features as received from the API.

    Returns:
        Dict[str, Any]: Prediction results including label, confidence, and probabilities.
    """
    if ml_model is None or ml_scaler is None or ml_label_encoder is None:
        raise RuntimeError("ML components not loaded. Call load_ml_components() first.")

    try:
        # Convert incoming dict to DataFrame, ensuring correct column order
        # Replace underscores in feature names to match original dataset if necessary
        # Example: "Destination_Port" from Pydantic becomes "Destination Port" for mapping to EXPECTED_FEATURES_ORDER
        # However, it's better if your Pydantic model uses the exact names of EXPECTED_FEATURES_ORDER.
        
        # Ensure the input dictionary keys match the EXPECTED_FEATURES_ORDER after any necessary transformation
        # Assuming FeaturesInput in schemas already handles naming conversions (e.g., snake_case to original)
        
        # A forma mais segura é construir o DataFrame a partir de uma lista de valores
        # na ordem correta, usando o EXPECTED_FEATURES_ORDER.
        
        # Construir um dicionário alinhado com EXPECTED_FEATURES_ORDER
        ordered_features = {
            col_name: features_data.get(col_name.replace(' ', '_').replace('/', '_').replace('.', '_')) 
            for col_name in EXPECTED_FEATURES_ORDER
        }

        # Converter para DataFrame. O pandas manterá a ordem das colunas
        input_df = pd.DataFrame([ordered_features])
        
        # Tratamento de valores especiais (NaN/Infinity) nos dados de entrada,
        # exatamente como foi feito no treinamento.
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        input_df.fillna(0, inplace=True)


        # Scale the input features
        X_processed = ml_scaler.transform(input_df)

        # Make prediction
        prediction_numeric = ml_model.predict(X_processed)[0]
        prediction_proba = ml_model.predict_proba(X_processed)[0].tolist()

        # Convert numeric prediction back to text label
        predicted_label = ml_label_encoder.inverse_transform([prediction_numeric])[0]
        confidence = prediction_proba[prediction_numeric if prediction_numeric < len(prediction_proba) else 0] # Avoid index error if prediction is somehow out of bounds

        return {
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "prediction_probabilities": {ml_label_encoder.classes_[i]: prob for i, prob in enumerate(prediction_proba)},
            "is_attack": bool(predicted_label == 'ATTACK'),
            "message": "Analysis complete." if predicted_label != 'ATTACK' else f"Potential DDoS attack detected with {round(confidence*100, 2)}% confidence."
        }
    except Exception as e:
        print(f"Error during ML prediction: {e}")
        raise # Re-raise the exception to be handled by the API route