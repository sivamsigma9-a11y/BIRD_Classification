# CNN Bird Classifier

This project implements a Convolutional Neural Network (CNN) to classify images as "Bird" or "No Bird". It includes functionality for training, evaluation, stress testing, and visual interpretability through feature map visualization.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd cnn-bird-classifier-analysis
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\Activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Training the Model

Train the model using the dataset in the `dataset/` directory. This script automatically logs experiments to MLflow.

```bash
python src/train.py
```

### 2. Evaluating the Model

Evaluate the model's performance on the validation set.

```bash
python src/evaluate.py
```

### 3. Prediction & Visualization

Run a prediction on a single image. This script will output the classification probability and **automatically save visualization of the convolutional filters (feature maps)**.

```bash
python src/predict.py <path_to_image>
```

**Example:**
```bash
python src/predict.py "path/to/image.jpg"
```

**Output:**
-   **Prediction:** Bird / No Bird (Probability)
-   **Feature Maps:** Saved to the `prediction_filters/` directory as `conv1.png`, `conv2.png`, and `conv3.png`. These images show what the model is "seeing" at different layers.

### 4. Stress Testing

Test the model's robustness against noise and brightness variations.

```bash
python src/stress_test.py <path_to_image>
```

The results and visualizations will be saved in the `filter_maps/` directory.

## MLflow Integration

Experiments, including parameters and metrics, are tracked using MLflow.

### Viewing Results

To view the MLflow UI and inspect your training runs:

**Option 1: Run Locally**

1.  Open your terminal/PowerShell.
2.  Run the MLflow UI command:
    ```bash
    python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
    ```
3.  Open your browser to [http://localhost:5000](http://localhost:5000).

**Option 2: Run with Docker**

1.  Ensure Docker is running.
2.  Run Docker Compose:
    ```bash
    docker-compose up
    ```
3.  Open your browser to [http://localhost:5000](http://localhost:5000).

**Note:** You do not need to run the MLflow UI to train the model. Logging happens automatically during training.