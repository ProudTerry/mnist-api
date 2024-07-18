from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = FastAPI()


@app.get("/")
async def read_root():

    return {
        "message": "Welcome to the MNIST digit classification API!",
        "instructions": {
            "POST /predict/": "Upload a grayscale image of a handwritten digit (28x28 pixels) to get the predicted class."
        }
    }


# Load the model
model = load_model('mnist_cnn_model.keras')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert('L').resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return {"predicted_class": int(predicted_class)}


