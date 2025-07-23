import pandas as pd
import os


def load_cicads_csv_data(directory_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Reads all CSV files from a given directory path and its subdirectories,
    and concatenates them into a single Pandas DataFrame.

    Handles common loading issues like encoding errors and malformed lines.
    Strips whitespace from column names for consistency.

    Args:
        directory_path (str): The root path to the directory containing the CSV files.
        encoding (str): The primary encoding to try for CSV files (default is 'utf-8').

    Returns:
        pd.DataFrame: A combined Pandas DataFrame from all loaded CSVs,
                      or an empty DataFrame if no CSVs are found or loaded successfully.
    """
    all_dfs = []
    print(f"Searching for CSV files in and under: {directory_path}")

    # Use os.walk to traverse directories and subdirectories
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.csv'):
                full_file_path = os.path.join(root, filename)
                print(f"  Loading {full_file_path}...")
                
                df_temp = None # Initialize df_temp to None for error handling

                try:
                    df_temp = pd.read_csv(full_file_path, encoding=encoding)
                except UnicodeDecodeError:
                    print(f"    UnicodeDecodeError for {filename}. Trying 'latin1' encoding...")
                    try:
                        df_temp = pd.read_csv(full_file_path, encoding='latin1')
                    except Exception as e:
                        print(f"      Failed with 'latin1' too for {filename}: {e}")
                except pd.errors.ParserError as e:
                    print(f"    ParserError for {filename}: {e}. Trying with 'on_bad_lines=skip'...")
                    try:
                        df_temp = pd.read_csv(full_file_path, encoding=encoding, on_bad_lines='skip')
                    except UnicodeDecodeError:
                         print(f"      UnicodeDecodeError with 'on_bad_lines=skip'. Trying 'latin1'...")
                         try:
                             df_temp = pd.read_csv(full_file_path, encoding='latin1', on_bad_lines='skip')
                         except Exception as e_inner:
                             print(f"        Failed with 'latin1' and skip too for {filename}: {e_inner}")
                    except Exception as e_inner:
                        print(f"      Failed to load {filename} even with 'on_bad_lines=skip': {e_inner}")
                except Exception as e:
                    print(f"    Unexpected error loading {filename}: {e}")

                if df_temp is not None:
                    # Strip whitespace from column names immediately upon loading for consistency
                    df_temp.columns = df_temp.columns.str.strip()
                    all_dfs.append(df_temp)
                    print(f"  {filename} loaded successfully with {df_temp.shape[0]} rows.")
                else:
                    print(f"  Skipping {filename} due to persistent loading errors.")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\nAll CSV datasets combined successfully! Final shape: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns.")
        
        # [cite_start]Checking for 'Label' column, common in CICIDS datasets [cite: 1]
        if 'Label' in combined_df.columns:
            print("\nCombined Dataset Label Counts:")
            print(combined_df['Label'].value_counts())
        # [cite_start]Also check for ' Label' with a leading space, common in some CICIDS CSVs [cite: 1]
        elif ' Label' in combined_df.columns:
            print("\nCombined Dataset ' Label' Counts:")
            print(combined_df[' Label'].value_counts())
        else:
            print("Label column (e.g., 'Label', ' Label') not found. Please check the exact column name.")

        print("\nChecking for missing values (NaN) in the combined dataset:")
        # [cite_start]Display columns with missing values and their counts [cite: 1]
        print(combined_df.isnull().sum()[combined_df.isnull().sum() > 0])
        return combined_df
    else:
        print("No CSV files found or none could be loaded successfully from the specified directory.")
        return pd.DataFrame() # Return an empty DataFrame if no data is loaded

# --- Example of Usage (for local testing of data_loader.py) ---
if __name__ == "__main__":
    # Adjust these paths relative to where you run this script from.
    # If your project root is 'ddos-detection-ml' and this script is in 'src/data_processing/'
    # then you need to go up two levels to reach 'datasets/'.
    
    # Path to MachineLearningCSV/MachineLearningCVE/
    # This directory contains subfolders like 'Monday-WorkingHours', 'Tuesday-WorkingHours', etc.
    # which in turn contain the actual CSV files.
    ML_CVE_PATH = '../../datasets/MachineLearningCSV/MachineLearningCVE/'
    
    # Path to GeneratedLabelledFlows/TrafficLabelling/
    # This directory also typically contains subfolders with CSV files.
    TRAFFIC_LABELLING_PATH = '../../datasets/GeneratedLabelledFlows/TrafficLabelling/'

    print("--- Loading MachineLearningCVE Dataset ---")
    df_ml_cve = load_cicads_csv_data(ML_CVE_PATH)

    if not df_ml_cve.empty:
        print("\nMachineLearningCVE Dataset Loaded (Head):")
        print(df_ml_cve.head())
        print("\nMachineLearningCVE Dataset Info:")
        df_ml_cve.info()
    else:
        print("MachineLearningCVE dataset could not be loaded or is empty.")


    print("\n--- Loading GeneratedLabelledFlows Dataset ---")
    df_traffic_labelling = load_cicads_csv_data(TRAFFIC_LABELLING_PATH)

    if not df_traffic_labelling.empty:
        print("\nTrafficLabelling Dataset Loaded (Head):")
        print(df_traffic_labelling.head())
        print("\nTrafficLabelling Dataset Info:")
        df_traffic_labelling.info()
    else:
        print("TrafficLabelling dataset could not be loaded or is empty.")
        
    print("\nData loading phase complete.")