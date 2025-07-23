import pandas as pd
import os


def leituraParsingDataset(diretorio: str) -> pd.DataFrame:
    conjunto_datasets = []

    print(f'Buscando arquivos CSV na pasta: {diretorio}')

    for nomeArquivo in os.listdir(diretorio):  # pode iterar por todos os arquivos naquela pasta
        if nomeArquivo.endswith('.csv'):
            dir = os.path.join(diretorio, nomeArquivo)
            print(f"Carregando {nomeArquivo}...")
            try:
                df_temp = pd.read_csv(dir, encoding='utf-8') 
                conjunto_datasets.append(df_temp)
                # print(f"  {nomeArquivo} carregado com {df_temp.shape[0]} linhas.")
            except pd.errors.ParserError as e:
                print(f"Erro de Parser ao carregar {nomeArquivo}: {e}. Tentando com outra codificação ou skip_bad_lines.")
                try:
                    df_temp = pd.read_csv(dir, encoding='latin1', on_bad_lines='skip')
                    conjunto_datasets.append(df_temp)
                    print(f"  {nomeArquivo} carregado com latin1 e ignorando linhas ruins, {df_temp.shape[0]} linhas.")
                except Exception as e_inner:
                    print(f"Falha total ao carregar {nomeArquivo}: {e_inner}")
            except Exception as e:
                print(f"Erro inesperado ao carregar {nomeArquivo}: {e}")

    # print(conjunto_datasets)

    if conjunto_datasets:
        df_combinados = pd.concat(conjunto_datasets, ignore_index=True)
        print("\nDatasets Machine Learning combinados com sucesso!")
        print(f"Dataset combinado final: {df_combinados.shape[0]} linhas, {df_combinados.shape[1]} colunas")
        if 'Label' in df_combinados.columns:
            print(df_combinados['Label'].value_counts())
        elif ' Label' in df_combinados.columns:
            print(df_combinados[' Label'].value_counts())
        else:
            print("Coluna de rótulo (ex: 'Label', ' Label') não encontrada. Verifique o nome exato da coluna.")

        print("\nVerificando valores ausentes (NaN) no dataset combinado:")
        print(df_combinados.isnull().sum()[df_combinados.isnull().sum() > 0])
        return df_combinados
    else:
        print("Nenhum arquivo CSV encontrado na pasta especificada ou nenhum pôde ser carregado.")
        return None

dirMachineLearning = 'datasets/MachineLearningCSV/MachineLearningCVE/'
conjuntoMLdfs = leituraParsingDataset(dirMachineLearning)  # todos os datasets da pasta machine learning

dirGenLabelledFlows = 'datasets/GeneratedLabelledFlows/TrafficLabelling'
conjuntoTrafficLabelling = leituraParsingDataset(dirGenLabelledFlows)
