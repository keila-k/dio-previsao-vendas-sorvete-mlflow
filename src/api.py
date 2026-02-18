from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Gelato Mágico - Previsão de Vendas")

class Input(BaseModel):
    temperature: float

# Ajuste: esse model_uri pega o último modelo do run mais recente do experimento.
# Para portfólio, simplificamos: usar o "latest" do diretório mlruns local pode variar.
# Alternativa robusta: você registrar em "Model Registry" (Azure ML/MLflow Tracking server).
model = None

@app.on_event("startup")
def load_model():
    global model
    # carrega o último modelo salvo no artifact local do run MAIS RECENTE do experimento
    # Para não complicar, você pode apontar manualmente depois de rodar o treino,
    # mas aqui vamos tentar carregar o "model" mais recente via busca no tracking local.
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("gelato-magico")
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise RuntimeError("Nenhum run encontrado. Execute: python src/train.py")
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)

@app.post("/predict")
def predict(inp: Input):
    df = pd.DataFrame([{"temperature": inp.temperature}])
    pred = model.predict(df)[0]
    return {"predicted_sales": float(pred)}
