import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, List


LABEL_COLUMN_NAME_CANDIDATES = ['Label', ' Label']
BENIGN_LABEL = 'BENIGN'  # rotulo para trafego normal

def clean_and_prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Performs data cleaning, feature selection, label encoding, and scaling.

    Args:
        df (pd.DataFrame): The raw DataFrame loaded from CICADS CSVs.

    Returns:
        Tuple[pd.DataFrame, pd.Series, LabelEncoder]: 
            - X: DataFrame with processed features.
            - y: Series with numerical labels.
            - label_encoder: The fitted LabelEncoder object (useful for inverse_transform).
    """
    print("\n--- Starting Data Cleaning and Feature Engineering ---")

    # 1. Tratar valores ausentes e infinitos
    # CICIDS datasets often contain infinity values which pandas treats as numbers.
    # Convert them to NaN first, then fill NaNs.
    print("Handling infinity values...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    print("Checking for missing values after handling infinities:")
    missing_before_fill = df.isnull().sum()
    print(missing_before_fill[missing_before_fill > 0])

    # Fill NaN values. For network flow data, 0 is often a sensible default
    # if a value is truly absent (e.g., no bytes in a flow).
    print("Filling missing values with 0...")
    df.fillna(0, inplace=True)
    print("Missing values after filling:", df.isnull().sum().sum()) # Should be 0

    # 2. Limpeza e Codificação dos Rótulos (Label)
    label_column = None
    for col_name in LABEL_COLUMN_NAME_CANDIDATES:
        if col_name in df.columns:
            label_column = col_name
            break
    
    if label_column is None:
        raise ValueError(f"Label column not found. Tried: {LABEL_COLUMN_NAME_CANDIDATES}. Available columns: {df.columns.tolist()}")

    print(f"Using '{label_column}' as the label column.")
    print("Original Label counts:")
    print(df[label_column].value_counts())

    # Ensure labels are strings and strip any whitespace
    df[label_column] = df[label_column].astype(str).str.strip()

    # Consolidate all attack types into a single 'ATTACK' label
    # This simplifies the problem to binary classification (BENIGN vs ATTACK)
    df['processed_label'] = df[label_column].apply(lambda x: 'ATTACK' if x != BENIGN_LABEL else BENIGN_LABEL)
    
    print("\nProcessed Label counts (BENIGN vs ATTACK):")
    print(df['processed_label'].value_counts())

    # Encode labels to numerical values (e.g., BENIGN=0, ATTACK=1)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['processed_label'])
    print(f"Label encoding mapping: {list(label_encoder.classes_)} -> {np.unique(y).tolist()}")

    # 3. Seleção de Features
    # Drop the original label column and the temporary processed_label column
    # Also, drop any columns that are clearly identifiers or irrelevant,
    # or that might have constant values (zero variance).
    # Some CICIDS columns might be duplicated or less useful, you might identify more later.
    columns_to_drop = [label_column, 'processed_label']
    
    # Drop columns that are entirely constant (zero variance)
    # This can happen after filling NaNs with 0 if a feature was always 0 or NaN.
    # It also helps remove features that provide no information.
    print("\nChecking for and dropping constant columns (zero variance)...")
    constant_columns = [col for col in df.columns if df[col].nunique() == 1 and col not in columns_to_drop]
    if constant_columns:
        print(f"Dropping constant columns: {constant_columns}")
        columns_to_drop.extend(constant_columns)
    else:
        print("No constant columns found to drop.")

    # Drop any non-numeric columns that remain and are not intended to be features
    # (e.g., if there's a 'Timestamp' column that hasn't been handled)
    # Ensure all remaining features are numeric before scaling
    numeric_features_df = df.drop(columns=columns_to_drop, errors='ignore').select_dtypes(include=[np.number])
    
    # Get the list of actual features used after all processing
    feature_columns = numeric_features_df.columns.tolist()
    X = numeric_features_df

    print(f"\nSelected {X.shape[1]} features for training.")
    print("Example features (first 5 rows):")
    print(X.head())

    # 4. Escalonamento das Features (Standardization)
    # StandardScaler transforms features to have a mean of 0 and standard deviation of 1.
    # This is crucial for many ML algorithms, especially those sensitive to feature scales (e.g., SVM, Neural Networks).
    print("\nScaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns) # Convert back to DataFrame
    print("Features scaled successfully.")
    print("Example scaled features (first 5 rows):")
    print(X.head())

    print("\n--- Data Cleaning and Feature Engineering Complete ---")
    return X, pd.Series(y, name='Target'), label_encoder

# --- Exemplo de uso (para teste local do feature_engineering.py) ---
if __name__ == "__main__":
    # Import the data loader function from the same package
    from data_loader import load_cicads_csv_data

    # Define paths to your datasets (adjust relative paths as needed)
    ML_CVE_PATH = '../../datasets/MachineLearningCSV/MachineLearningCVE/'
    TRAFFIC_LABELLING_PATH = '../../datasets/GeneratedLabelledFlows/TrafficLabelling/'

    # Load data using the data_loader
    print("Loading MachineLearningCVE data for feature engineering test...")
    raw_df_ml_cve = load_cicads_csv_data(ML_CVE_PATH)

    if not raw_df_ml_cve.empty:
        try:
            X_ml_cve, y_ml_cve, le_ml_cve = clean_and_prepare_data(raw_df_ml_cve)
            print("\nProcessed MachineLearningCVE Data (X shape):", X_ml_cve.shape)
            print("Processed MachineLearningCVE Labels (y shape):", y_ml_cve.shape)
            print("Label Encoder Classes:", le_ml_cve.classes_)
        except Exception as e:
            print(f"Error during feature engineering for MachineLearningCVE: {e}")
    else:
        print("Skipping feature engineering for MachineLearningCVE as data could not be loaded.")

    print("\nLoading GeneratedLabelledFlows data for feature engineering test...")
    raw_df_traffic_labelling = load_cicads_csv_data(TRAFFIC_LABELLING_PATH)

    if not raw_df_traffic_labelling.empty:
        try:
            X_traffic_labelling, y_traffic_labelling, le_traffic_labelling = clean_and_prepare_data(raw_df_traffic_labelling)
            print("\nProcessed TrafficLabelling Data (X shape):", X_traffic_labelling.shape)
            print("Processed TrafficLabelling Labels (y shape):", y_traffic_labelling.shape)
            print("Label Encoder Classes:", le_traffic_labelling.classes_)
        except Exception as e:
            print(f"Error during feature engineering for TrafficLabelling: {e}")
    else:
        print("Skipping feature engineering for TrafficLabelling as data could not be loaded.")

    print("\nFeature engineering testing complete.")
