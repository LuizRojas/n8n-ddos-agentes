# src/api/main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager # Para eventos de startup/shutdown
from .routes import ddos_detection # Importa o roteador
from .services.ml_service import load_ml_components # Importa a função de carregamento de modelos

# Gerenciador de contexto para eventos de inicialização e desligamento da API
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Evento de startup: Carregar os componentes de ML
    print("API startup: Loading ML components...")
    load_ml_components()
    yield
    # Evento de shutdown (se houver algo para limpar, como fechar conexões)
    print("API shutdown: Cleaning up...")


app = FastAPI(
    title="DDoS Detection API",
    description="API for classifying network traffic as DDoS or Normal using Machine Learning.",
    version="1.0.0",
    lifespan=lifespan # Associa o gerenciador de contexto à aplicação
)

# Incluir as rotas definidas em ddos_detection.py
app.include_router(ddos_detection.router, prefix="/api/v1", tags=["DDoS Prediction"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the DDoS Detection API! Visit /docs for API documentation."}

# Para executar: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# Certifique-se de estar na raiz do projeto (ddos-detection-ml/)