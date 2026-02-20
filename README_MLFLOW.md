# How to Run MLflow

## Option 1: Run Locally (Recommended)

1.  **Open PowerShell**.
2.  Run the start script:
    ```powershell
    python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
    ```
3.  Open your browser to [http://localhost:5000](http://localhost:5000).

## Option 2: Run with Docker

1.  **Open PowerShell**.
2.  Run Docker Compose:
    ```powershell
    docker-compose up
    ```
3.  Open your browser to [http://localhost:5000](http://localhost:5000).

## How to Log Data

You do **not** need to "run" MLflow to train your model. The logging happens automatically when you run the scripts:

- **Train**: `python src/train.py`
- **Evaluate**: `python src/evaluate.py`

The results are stored in `mlflow.db` and viewed using the commands above.
