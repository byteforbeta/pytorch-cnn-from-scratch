# Custom PyTorch CNN & FastAPI Inference for Fashion-MNIST

A complete, production-ready Machine Learning pipeline built from scratch. This repository demonstrates the end-to-end lifecycle of a computer vision project: from designing a custom Convolutional Neural Network (CNN) in PyTorch to serving the trained model through an asynchronous inference API using FastAPI.

## Key Features
* **Custom Architecture:** Built a multi-layer CNN with Max Pooling and Dropout regularization to prevent overfitting.
* **Production-Grade Training:** Implemented deterministic seeding, dynamic YAML configuration, and multi-worker data loading.
* **High Accuracy:** Achieved **92.66% accuracy** on the unseen Fashion-MNIST test dataset.
* **API Integration:** Wrapped the trained model in a local FastAPI endpoint with Pydantic validation and normalized tensor preprocessing.

## Project Structure

```text
├── data/                   # Automatically downloads the Fashion-MNIST dataset here
├── saved_models/           # Stores the best .pth model weights
├── src/
│   ├── app.py              # FastAPI application and inference endpoint
│   ├── config.yaml         # Single source of truth for hyperparameters
│   ├── dataset.py          # Data loaders and transform pipelines
│   ├── model.py            # The CNN architecture class
│   └── train.py            # Main training and evaluation loop
├── .gitignore
├── README.md
└── requirements.txt        # Python dependencies
```

## Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/YOUR_USERNAME/fashion-mnist-cnn-pytorch.git](https://github.com/YOUR_USERNAME/fashion-mnist-cnn-pytorch.git)
cd pytorch-cnn-from-scratch
```

2. **Create a virtual environment and install dependencies:**
```bash
python -m venv venv
# On Windows use: venv\Scripts\activate
source venv/bin/activate  
pip install -r requirements.txt
```

## 🧠 Training the Model

All hyperparameters (learning rate, batch size, optimizer, etc.) are decoupled from the core code. You can easily modify them in `src/config.yaml`.

To start the training loop:
```bash
python src/train.py
```
*Note: The script automatically detects and utilizes NVIDIA GPUs (CUDA) if available, falling back to CPU if necessary.*

## 🌐 Running the Inference API

Once the model is trained and the `.pth` weights are saved in `saved_models/`, you can spin up the FastAPI server to run predictions on new images.

1. **Start the server:**
```bash
uvicorn src.app:app --reload
```

2. **Test the Endpoint:**
Open your browser and navigate to `http://127.0.0.1:8000/docs`. You can use the built-in Swagger UI to upload an image of a clothing item directly to the `/predict` endpoint and receive a JSON response with the model's prediction and confidence score.

## 📊 Results
* **Training Loss:** ~0.163
* **Test Accuracy:** 92.66%