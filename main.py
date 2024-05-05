from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import numpy as np
import requests
import uvicorn

# Initialize FastAPI application
app = FastAPI()

# Allow requests from specified origins
origins = [
    'http://localhost',
    'http://localhost:3000'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Define endpoint for model predictions
endpoint = 'http://localhost:8605/v1/models/dr-tomato-dev-py39/labels/production:predict'

# Define class names for model predictions
class_names = ['Bacterial-spot', 'Early-blight', 'Healthy', 'Late-blight',
               'Leaf-mold', 'Mosaic-virus', 'Septoria-leaf-spot', 'Yellow-leaf-curl-virus']

# Test connection to the server
@app.get('/ping')
async def ping():
    """
    Test endpoint to check if the server is ready.
    """
    return 'Ready!'

# Predict image class
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the class of an uploaded image.

    Parameters:
    - file: UploadFile - the image file to be predicted.

    Returns:
    - dict: Predicted class and confidence.
    """
    file_bytes = await file.read()  # Read file bytes
    img = Image.open(BytesIO(file_bytes))  # Open image from bytes
    img_array = np.array(img)  # Convert image to numpy array
    img_batch = np.expand_dims(img_array, axis=0)  # Create image batch for prediction

    # Prepare data for prediction
    json_data = {'instances': img_batch.tolist()}
    
    # Make prediction request to the model endpoint
    response = requests.post(endpoint, json=json_data)
    pred = response.json()['predictions'][0]  # Get prediction results

    # Get predicted class and confidence
    pred_class = class_names[np.argmax(pred)]
    pred_conf = float(np.max(pred))

    return {'pred_class': pred_class, 'pred_conf': pred_conf}

# Run the FastAPI application using uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
