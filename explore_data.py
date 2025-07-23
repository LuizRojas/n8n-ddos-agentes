import pandas as pd
import os


dirMachineLearning = 'datasets/MachineLearningCSV/MachineLearningCVE/'
conjuntoMLdfs = []  # todos os datasets da pasta machine learning

print(f'Buscando arquivos CSV na pasta: {dirMachineLearning}')

for nomeArquivo in os.listdir(dirMachineLearning):  # pode iterar por todos os arquivos naquela pasta
    if nomeArquivo.endswith('.csv'):
        diretorio = os.path.join(dirMachineLearning, nomeArquivo)
        print(f"Carregando {nomeArquivo}...")
        try:
            df_temp = pd.read_csv(diretorio, encoding='utf-8') 
            conjuntoMLdfs.append(df_temp)
            print(f"  {nomeArquivo} carregado com {df_temp.shape[0]} linhas.")
        except pd.errors.ParserError as e:
            print(f"Erro de Parser ao carregar {nomeArquivo}: {e}. Tentando com outra codificação ou skip_bad_lines.")
            try:
                df_temp = pd.read_csv(diretorio, encoding='latin1', on_bad_lines='skip')
                conjuntoMLdfs.append(df_temp)
                print(f"  {nomeArquivo} carregado com latin1 e ignorando linhas ruins, {df_temp.shape[0]} linhas.")
            except Exception as e_inner:
                print(f"Falha total ao carregar {nomeArquivo}: {e_inner}")
        except Exception as e:
            print(f"Erro inesperado ao carregar {nomeArquivo}: {e}")

print(conjuntoMLdfs)