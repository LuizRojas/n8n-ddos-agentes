# src/ml_model/trainer.py

import joblib
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.data_processing.data_loader import load_cicads_csv_data
from src.data_processing.feature_engineering import clean_and_prepare_data, BENIGN_LABEL # Importe BENIGN_LABEL

# --- Configuração de Caminhos para os Datasets e Modelos ---
# Estes caminhos são relativos à RAIZ DO PROJETO,
# presumindo que você executará este script da raiz do projeto (ex: python src/ml_model/trainer.py).

ML_CVE_PATH = 'datasets/MachineLearningCSV/MachineLearningCVE/'
TRAFFIC_LABELLING_PATH = 'datasets/GeneratedLabelledFlows/TrafficLabelling/'

MODELS_DIR = 'models/'
MODEL_FILENAME = 'ddos_classifier_rf_model.joblib'
SCALER_FILENAME = 'ddos_classifier_scaler.joblib'
LABEL_ENCODER_FILENAME = 'ddos_classifier_label_encoder.joblib'
FEATURE_ORDER_FILENAME = 'feature_columns_order.joblib' # Para salvar a ordem das features


def train_and_save_model():
    """
    Loads, preprocesses, trains, and saves the ML model and its components.
    """
    # 1. Carregar os dados brutos de ambos os datasets CICADS
    print("--- Loading CICADS Datasets ---")
    raw_df_ml_cve = load_cicads_csv_data(ML_CVE_PATH)
    raw_df_traffic_labelling = load_cicads_csv_data(TRAFFIC_LABELLING_PATH)

    if raw_df_ml_cve.empty and raw_df_traffic_labelling.empty:
        print("No data loaded for training. Aborting model training.")
        return

    # Combine os DataFrames carregados (se ambos existirem e não estiverem vazios)
    combined_raw_df = pd.DataFrame()
    if not raw_df_ml_cve.empty:
        combined_raw_df = pd.concat([combined_raw_df, raw_df_ml_cve], ignore_index=True)
    if not raw_df_traffic_labelling.empty:
        combined_raw_df = pd.concat([combined_raw_df, raw_df_traffic_labelling], ignore_index=True)

    if combined_raw_df.empty:
        print("Combined DataFrame is empty after loading. Aborting training.")
        return

    print(f"\nCombined Raw DataFrame shape: {combined_raw_df.shape}")

    # 2. Pré-processar e engenharia de features
    # clean_and_prepare_data precisa retornar o scaler para que possamos salvá-lo
    # Modifique src/data_processing/feature_engineering.py para retornar o scaler também!
    # from src.data_processing.feature_engineering import clean_and_prepare_data, BENIGN_LABEL, StandardScaler # <- StandardScaler para retornar a instância
    # (Se feature_engineering.py não retornar o scaler, você pode inicializar e fitar um novo aqui, mas não é o ideal)

    # Para que clean_and_prepare_data retorne o scaler, ele deve ter o retorno modificado para:
    # `return X, pd.Series(y, name='Target'), label_encoder, scaler`
    # E aqui:
    X, y_encoded, label_encoder, scaler = clean_and_prepare_data(combined_raw_df) # Assuming modified clean_and_prepare_data

    # 3. Divisão dos dados em treino e teste
    print("\n--- Splitting Data into Training and Testing Sets ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded) 

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # 4. Treinar o modelo
    print("\n--- Training RandomForestClassifier Model ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced') 
    # class_weight='balanced' é útil para datasets desbalanceados (muito mais tráfego normal que ataque)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Avaliar o modelo
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred):.4f}")

    # 6. Exportar os componentes do modelo
    print("\n--- Saving ML Components ---")
    # Criar o diretório de modelos se não existir
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODELS_DIR, MODEL_FILENAME))
    print(f"Model saved to: {os.path.join(MODELS_DIR, MODEL_FILENAME)}")

    joblib.dump(scaler, os.path.join(MODELS_DIR, SCALER_FILENAME))
    print(f"Scaler saved to: {os.path.join(MODELS_DIR, SCALER_FILENAME)}")

    joblib.dump(label_encoder, os.path.join(MODELS_DIR, LABEL_ENCODER_FILENAME))
    print(f"Label Encoder saved to: {os.path.join(MODELS_DIR, LABEL_ENCODER_FILENAME)}")

    # Salvar a ordem das features para uso na API
    joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, FEATURE_ORDER_FILENAME))
    print(X.columns.tolist())
    print(f"Feature order saved to: {os.path.join(MODELS_DIR, FEATURE_ORDER_FILENAME)}")

    print("\nModel training and saving process complete!")


# Exemplo de como modificar 'feature_engineering.py' para retornar o scaler
# no arquivo src/data_processing/feature_engineering.py, altere a assinatura da função:
# def clean_and_prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, StandardScaler]:
# E no final da função, onde você tem `return X, pd.Series(y, name='Target'), label_encoder`, mude para:
# `return X, pd.Series(y, name='Target'), label_encoder, scaler`

if __name__ == "__main__":
    # Esta linha irá iniciar todo o processo de treinamento e salvamento do modelo
    train_and_save_model()