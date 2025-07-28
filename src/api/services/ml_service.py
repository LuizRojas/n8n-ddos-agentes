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
FEATURE_ORDER_PATH = 'models/feature_columns_order.joblib' # <<-- NOVO: Caminho para a ordem das features


# Variáveis globais para armazenar os componentes de ML
ml_model = None
ml_scaler = None
ml_label_encoder = None
ml_feature_order = None # <<-- NOVO: Variável para armazenar a ordem das features

def load_ml_components():
    """
    Loads the trained ML model, scaler, label encoder, and feature order from disk.
    This function should be called once at API startup.
    """
    global ml_model, ml_scaler, ml_label_encoder, ml_feature_order
    
    # Verifica se os componentes já foram carregados
    if ml_model is not None and ml_scaler is not None and ml_label_encoder is not None and ml_feature_order is not None:
        print("ML components already loaded.")
        return 

    print("Loading ML model, scaler, label encoder, and feature order...") # <<-- Mensagem atualizada
    try:
        # Verifica a existência de todos os arquivos necessários
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {os.path.abspath(MODEL_PATH)}")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found: {os.path.abspath(SCALER_PATH)}")
        if not os.path.exists(LABEL_ENCODER_PATH):
            raise FileNotFoundError(f"Label Encoder file not found: {os.path.abspath(LABEL_ENCODER_PATH)}")
        if not os.path.exists(FEATURE_ORDER_PATH): # <<-- NOVO: Verifica o arquivo da ordem das features
            raise FileNotFoundError(f"Feature order file not found: {os.path.abspath(FEATURE_ORDER_PATH)}")

        # Carrega os componentes
        ml_model = joblib.load(MODEL_PATH)
        ml_scaler = joblib.load(SCALER_PATH)
        ml_label_encoder = joblib.load(LABEL_ENCODER_PATH)
        ml_feature_order = joblib.load(FEATURE_ORDER_PATH) # <<-- CRÍTICO: Carrega a ordem das features aqui
        
        print("ML components loaded successfully!")
        print(f"Loaded Label Encoder classes: {ml_label_encoder.classes_}")
        print(f"Number of expected features: {len(ml_feature_order)}")
        # print(f"Expected features (first 5): {ml_feature_order[:5]}...") # Para depuração

    except Exception as e:
        print(f"Error loading ML components: {e}")
        raise # Rethrow para impedir que a API inicie sem os modelos

def predict_ddos_attack(features_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Receives raw features (from API request), preprocesses them, and makes a prediction.

    Args:
        features_data (Dict[str, Any]): A dictionary of features as received from the API.

    Returns:
        Dict[str, Any]: Prediction results including label, confidence, probabilities,
                        and 'real_client_ip_for_alert' for context.
    """
    # Verifica se todos os componentes de ML foram carregados
    if ml_model is None or ml_scaler is None or ml_label_encoder is None or ml_feature_order is None: # <<-- Inclui ml_feature_order
        raise RuntimeError("ML components not loaded. Call load_ml_components() first.")

    try:
        # Extrai o IP real do cliente que foi injetado pelo n8n
        client_real_ip = features_data.get("client_real_ip", "UNKNOWN_IP_FROM_API") # <<-- NOVO: Captura o IP real

        # --- Constrói o DataFrame de entrada para o modelo (APENAS com as features esperadas) ---
        ordered_features = {}
        # Itera pela lista de nomes de features que o modelo ESPERA (carregada de ml_feature_order)
        for col_name in ml_feature_order: 
            # Converte o nome da feature (que está no formato original do treino, ex: "Destination Port")
            # para o formato que a Pydantic/JSON usa (ex: "Destination_Port")
            pydantic_key = col_name.replace(' ', '_').replace('/', '_').replace('.', '_')
            
            # Pega o valor do dicionário de entrada (features_data)
            # Se o campo não existir na entrada (o que pode acontecer se o n8n não o enviou), usa 0 como default.
            # É CRÍTICO que o valor default (0) seja compatível com a feature.
            value = features_data.get(pydantic_key, 0) 

            # Tratamento de valores nulos/infinitos na entrada da API, conforme feito no treino
            if value is None:
                value = np.nan
            elif isinstance(value, (int, float)):
                if value == float('inf') or value == -float('inf'):
                    value = np.nan
            
            ordered_features[col_name] = value # Guarda o valor com o NOME ORIGINAL DA FEATURE (como o modelo espera)

        # Converte para DataFrame. O Pandas manterá a ordem das colunas, essencial para scikit-learn
        input_df = pd.DataFrame([ordered_features])
        
        # Tratamento de valores especiais (NaN/Infinity) novamente, por segurança.
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        input_df.fillna(0, inplace=True) # Preenche NaNs com 0, como no treino


        # Escala as features de entrada usando o scaler carregado
        X_processed = ml_scaler.transform(input_df)

        # Faz a predição
        prediction_numeric = ml_model.predict(X_processed)[0]
        prediction_proba = ml_model.predict_proba(X_processed)[0].tolist()

        # Converte a predição numérica de volta para o rótulo de texto
        predicted_label = ml_label_encoder.inverse_transform([prediction_numeric])[0]
        confidence = prediction_proba[prediction_numeric if prediction_numeric < len(prediction_proba) else 0]
        
        # Constrói a resposta da API
        response = {
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "prediction_probabilities": {ml_label_encoder.classes_[i]: prob for i, prob in enumerate(prediction_proba)},
            "is_attack": bool(predicted_label == 'ATTACK'),
            "message": "Analysis complete." if predicted_label != 'ATTACK' else f"Potential DDoS attack detected with {round(confidence*100, 2)}% confidence.",
            "real_client_ip_for_alert": client_real_ip # <<-- INCLUI O IP REAL NA RESPOSTA DA API
        }
        
        return response

    except Exception as e:
        print(f"Error during ML prediction: {e}")
        # Retorna um erro HTTP 500 com detalhes do erro para depuração no n8n
        raise # Isso re-lançará a exceção para o FastAPI tratar e retornar o 500