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
    Receives raw features (from API request) and makes a mock prediction.
    This bypasses the trained model to demonstrate the workflow.
    """
    try:
        # Captura o IP real do cliente que foi injetado pelo n8n
        client_real_ip = features_data.get("client_real_ip", "UNKNOWN_IP_FROM_API")

        # --- LÓGICA MOCK DE PREDIÇÃO ---
        # Para demonstrar a API fazendo uma "decisão", vamos usar uma regra simples
        # baseada na feature 'Total_Fwd_Packets', que o n8n vai mockar.

        # Pega o valor da feature do JSON de entrada.
        # O get() é usado para evitar erros caso a feature não esteja lá.
        total_fwd_packets = features_data.get("Total_Fwd_Packets", 0)

        # Define um limiar para a decisão da IA mockada.
        # Se o número de pacotes for maior que 10, consideramos como ataque.
        # Este é o nosso "padrão de ataque" simulado.
        mock_threshold = 10 

        if total_fwd_packets > mock_threshold:
            predicted_label = 'ATTACK'
            confidence = 0.95 # Alta confiança para o ataque simulado
            message = "Anomalia de tráfego detectada! Alto volume de requisições."
        else:
            predicted_label = 'BENIGN'
            confidence = 0.99 # Alta confiança para tráfego normal simulado
            message = "Tráfego analisado e considerado normal."

        # A API ainda retorna a estrutura completa de predição, mesmo com valores mockados.
        response = {
            "prediction": predicted_label,
            "confidence": confidence,
            "prediction_probabilities": {"ATTACK": confidence, "BENIGN": 1.0 - confidence},
            "is_attack": bool(predicted_label == 'ATTACK'),
            "message": message,
            "real_client_ip_for_alert": client_real_ip # Inclui o IP real na resposta da API
        }

        return response

    except Exception as e:
        print(f"Error during mock ML prediction: {e}")
        raise # Re-lança a exceção