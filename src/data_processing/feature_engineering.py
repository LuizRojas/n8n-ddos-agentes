# src/data_processing/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple

LABEL_COLUMN_NAME_CANDIDATES = ['Label', ' Label']
BENIGN_LABEL = 'BENIGN'

def clean_and_prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, StandardScaler]:
    """
    Gera features de alto nível a partir do dataset CICADS original,
    simulando o que seria extraído de logs HTTP para um modelo mais pragmático.
    """
    print("\n--- Starting HTTP Log-Based Feature Engineering ---")

    # 1. Limpeza e tratamento inicial
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df.columns = df.columns.str.strip() # Limpa nomes das colunas

    # 2. SELEÇÃO E GERAÇÃO DAS NOVAS FEATURES
    print("Generating new features based on HTTP log data...")

    # Primeiro, vamos criar a coluna de rótulo processada ANTES da seleção de features
    label_column = None
    for col_name in LABEL_COLUMN_NAME_CANDIDATES:
        if col_name in df.columns:
            label_column = col_name
            break
    
    if label_column is None:
        raise ValueError(f"Label column not found. Tried: {LABEL_COLUMN_NAME_CANDIDATES}. Available columns: {df.columns.tolist()}")

    df[label_column] = df[label_column].astype(str).str.strip()
    df['processed_label'] = df[label_column].apply(lambda x: 'ATTACK' if x != BENIGN_LABEL else BENIGN_LABEL)
    
    # 3. Codificação dos Rótulos (y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['processed_label'])

    # 4. Geração das features que simulamos a partir de logs HTTP
    # A lista de features deve corresponder à classe FeaturesInput do Pydantic
    feature_names_to_keep = [
        'Destination Port',
        'Flow Duration',
        'Total Fwd Packets',
        'Total Length of Fwd Packets',
        'SYN Flag Count',
        'RST Flag Count',
        'ACK Flag Count',
        'Source Port',
        'Protocol',
    ]

    # Crie um novo DataFrame X com apenas as features que queremos manter
    # O .copy() é usado para evitar warnings do Pandas sobre 'slice' em um DataFrame
    X = df[feature_names_to_keep].copy()

    # 5. Escalonamento das Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    print(f"\nGenerated {X.shape[1]} features for training:")
    print(X.columns.tolist())
    print("Features scaled successfully.")

    # Retorne X, y, o encoder e o scaler, na ordem correta.
    return X, pd.Series(y, name='Target'), label_encoder, scaler

if __name__ == "__main__":
    from data_loader import load_cicads_csv_data
    ML_CVE_PATH = '../../datasets/MachineLearningCSV/MachineLearningCVE/'
    raw_df = load_cicads_csv_data(ML_CVE_PATH)
    if not raw_df.empty:
        try:
            X, y, le, scaler = clean_and_prepare_data(raw_df)
            print("Feature engineering complete with new features!")
        except Exception as e:
            print(f"An error occurred during feature engineering: {e}")