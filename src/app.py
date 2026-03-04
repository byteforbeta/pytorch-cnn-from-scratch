import os
import io
import yaml
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torchvision.transforms as transforms
from src.model import CnnNet


cwd = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open(os.path.join(cwd, "src", "config.yaml"), 'r') as cf:
    config = yaml.load(cf, Loader=yaml.SafeLoader)

model = CnnNet(
    in_channels=config['model']['in_channels'], 
    hidden_size=config['model']['fc_hidden_size'], 
    num_classes=config['model']['num_classes'],
    dropout_rate=config['model']['dropout_rate']
).to(device)

model_path = os.path.join(cwd, "saved_models", "custom_cnn.pth")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

model.eval()

class_mapping = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Force image to 1 channel
    transforms.Resize((28, 28)),                 # Force image to 28x28 pixels
    transforms.ToTensor(),                       # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(mean=(0.2861,), std=(0.3530,)) 
])


app = FastAPI(title="Fashion-MNIST CNN API", description="Custom PyTorch CNN Inference API")


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        input_tensor = inference_transform(image)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Apply softmax to convert raw logits into actual percentages (0.0 to 1.0)
            confidence, predicted_index = torch.max(probabilities, 1) # Get the highest probability and its corresponding class index

        predicted_class = class_mapping[predicted_index.item()]  # Convert tensor values back to standard Python datatypes
        confidence_score = round(confidence.item() * 100, 2)

        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": f"{confidence_score}%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))