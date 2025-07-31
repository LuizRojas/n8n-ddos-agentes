# Mitigação de Ataques DDoS com Inteligência Artificial

## 1. Visão Geral do Projeto

Este projeto visa detectar ataques DDoS em tempo real utilizando algoritmos de **aprendizado de máquina**. A solução classifica requisições como **tráfego legítimo** ou **malicioso**, analisando os padrões dos logs HTTP. A arquitetura emprega **Docker**, **FastAPI**, **n8n** e serviços de proxy e web para simular um ambiente real, com foco em **portabilidade** e **automação da detecção**.

---

## 2. Arquitetura da Solução

O sistema é composto por quatro serviços principais, orquestrados com **Docker Compose**:

- **nginx (Proxy Reverso):** Porta de entrada do tráfego. Encaminha as requisições para o Apache e adiciona cabeçalhos como `X-Real-IP`.
- **apache-target (Servidor Web):** Simula o alvo do ataque. Os logs de acesso gerados alimentam o sistema de detecção.
- **ml_api (API de Machine Learning):** API FastAPI que carrega o modelo de IA treinado. Recebe features e retorna a predição (`BENIGN` ou `ATTACK`).
- **n8n (Workflow de Automação):** Orquestra todo o fluxo de leitura de logs, extração de features, predição e notificação (via Telegram).

---

## 3. Configuração e Instalação

### Pré-requisitos

- Docker e Docker Compose
- Python 3.x com bibliotecas listadas em `requirements.txt`

### Estrutura de Pastas

```
n8n-ddos-agentes/
├── src/
│   ├── api/
│   │   ├── ... (código da API)
│   │   ├── schemas/
│   │   │   └── prediction_schemas.py
│   │   └── services/
│   │       └── ml_service.py
│   ├── data_processing/
│   │   └── feature_engineering.py
│   └── ml_model/
│       └── trainer.py
├── datasets/
│   └── MachineLearningCSV/...
├── models/
├── nginx_conf/
│   ├── nginx.conf
│   └── default.conf
├── apache_conf/
│   └── httpd.conf
├── n8n_data/
├── nginx_logs/
├── apache_logs/
├── .env.example
├── requirements.txt
└── docker-compose.yaml
```

### Passos para Execução

1. Copie o `.env.example` para `.env` e preencha:

```env
N8N_BASIC_AUTH_USER=seu_usuario
N8N_BASIC_AUTH_PASSWORD=sua_senha
```

2. Inicie os serviços:

```bash
docker-compose up -d --build --force-recreate
```

---

## 4. Treinamento da Inteligência Artificial

O modelo de IA deve ser treinado localmente (fora do Docker). Para isso:

### Executar Treinamento

```bash
python -m src.ml_model.trainer
```

- **Modelo:** `RandomForestClassifier`
- **Dataset:** CIC-IDS-2023 (pasta `datasets/`)
- **Saídas:** `model.joblib`, `scaler`, `feature_columns_order.joblib`
- **Features utilizadas:** `request_count`, `error_rate`, `average_bytes_sent`

---

## 5. Workflow de Detecção no n8n

O n8n automatiza todo o fluxo de detecção com os seguintes nós:

1. **On Schedule** – Executa o fluxo a cada minuto.
2. **Read Binary File** – Lê `access.log` do Nginx.
3. **Code (JavaScript)** – Extrai features dos logs e insere o IP real do atacante.
4. **HTTP Request** – Envia dados para `http://ddos_ml_api:8000/api/v1/predict_ddos`.
5. **Telegram** – Envia alerta ao usuário com o status da predição.

---

## 6. Uso e Demonstração

### Simular Ataque

Na máquina Kali Linux, execute:

```bash
siege -c 50 -t 30S http://<IP_DO_HOST>/
```

### Executar o Workflow

- Acesse o n8n: [http://localhost:5678/](http://localhost:5678/)
- Execute manualmente ou aguarde o agendamento

### Resultado Esperado

- A IA analisará o padrão de requisições.
- Um alerta será enviado via Telegram com:
  - IP real (injetado)
  - Predição (`ATTACK` ou `BENIGN`)
  - Confiança da IA

---

## 7. Conclusão

Durante o desenvolvimento, os principais desafios incluíram:

- Obtenção do IP real do cliente em ambiente Docker
- Adaptação de features de rede (CIC-IDS) para dados de log HTTP
- Integração de múltiplos serviços de forma modular e em tempo real

O projeto valida a aplicabilidade de **Inteligência Artificial na cibersegurança**, entregando uma solução automatizada, portátil e funcional para mitigação de ataques DDoS.

---
