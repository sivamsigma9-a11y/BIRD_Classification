$dbPath = Join-Path $PSScriptRoot "mlflow.db"
python -m mlflow ui --backend-store-uri "sqlite:///$dbPath" --port 5000
