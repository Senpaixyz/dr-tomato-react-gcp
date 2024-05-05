## Leaf Disease Classifier (Mobile and Web Application)

Leaf Disease Classifier is a user-friendly mobile and web app designed to help tomato plant growers identify and treat diseases. This app is capable of recognizing the symptoms of various tomato plant diseases, including healthy tomato, early blight, late blight, bacterial spot, and yellow leaf curl. With this app, you can easily diagnose any of these issues your tomato plants may be facing, and get clear instructions on how to treat them effectively.Leaf Disease Classifier is a user-friendly mobile and web app designed to help tomato plant growers identify and treat diseases. This app is capable of recognizing the symptoms of various tomato plant diseases, including healthy tomato, early blight, late blight, bacterial spot, and yellow leaf curl. With this app, you can easily diagnose any of these issues your tomato plants may be facing, and get clear instructions on how to treat them effectively.

### Model 
The table presents evaluation metrics such as Precision, Recall, and F1-score for different labels in a classification task, likely related to plant diseases. Each label represents a distinct class, such as Bacterial-spot, Early-blight, Healthy, and so on. Precision measures the accuracy of positive predictions, with notable values like 100.00% for Mosaic-virus and Yellow-leaf-curl-virus, indicating near-perfect accuracy in these cases. Recall gauges the model's ability to capture actual positives, with Healthy achieving a perfect 100.00%. F1-score, the harmonic mean of Precision and Recall, balances these metrics; for instance, Early-blight achieves an impressive F1-score of 97.24%, demonstrating a strong balance between precision and recall. Overall, the table showcases the model's effectiveness in distinguishing between different classes, with varying degrees of precision, recall, and F1-scores highlighting its performance nuances across diverse disease categories.

| Label |Precision |	Recall |	F1 |
|----------|----------|:-------------:|------:|
Bacterial-spot |	100.00% |	99.08% |	99.54% |
Early-blight |	95.65% |	98.88% |	97.24% |
Healthy |	98.29% |	100.00% |	99.14% |
Late-blight |	95.33% |	98.08% |	96.68% |
Leaf-mold |	98.26% |	95.76% |	97.00% |
Mosaic-virus |	100.00 |%	100.00% |	100.00% |
Septoria-leaf-spot |	97.59% |	93.10% |	95.29% |
Yellow-leaf-curl-virus |	100.00% |	100.00% |	100.00% |

### Web Application Screenshots

Here you can drag and drop leaf image from your device" refers to a user interface (UI) feature that allows users to upload images by dragging them from their computer or mobile device and dropping them onto a designated area within the application

![Initial](https://raw.githubusercontent.com/Senpaixyz/dr-tomato-react-gcp/main/1686379204502.jpg)

After an image is uploaded, the backend initiates a series of processes starting with image processing to ensure compatibility for analysis. The extracted features from the image, such as patterns or color distributions indicative of disease states, are then used by a pre-trained machine learning model. This model predicts the most likely disease state or condition of the subject in the image, which could be plants or medical images. Finally, the predicted disease state or condition is communicated back to the user, providing valuable insights into the current health status or any potential issues detected in the uploaded image.

![Initial](https://raw.githubusercontent.com/Senpaixyz/dr-tomato-react-gcp/main/1686379228425.jpg)

Once the image analysis is complete, the model generates a predicted result based on the identified features, showcasing insights such as the detected disease state or other relevant information derived from the image. This prediction is then displayed to the user through the application interface, enabling informed decision-making and enhancing the user experience by providing actionable information.

![Initial](https://raw.githubusercontent.com/Senpaixyz/dr-tomato-react-gcp/main/1686379271842%20(1).jpg)

The predict function is a crucial component of a FastAPI application designed to handle POST requests for predicting the class of an uploaded image. It receives the image file as input, processes it to extract image data, and then sends this data to a specified model endpoint. Notably, the model endpoint is hosted on Google Cloud Platform (GCP) Lambda, serving as the backend for the image classification model. This setup enables seamless integration with a frontend application, which is built using the Flask framework for web development and React Native for mobile development. The Flask framework handles web requests, while React Native manages mobile app requests, both of which interact with the FastAPI application through HTTP requests. This architecture ensures efficient communication between the frontend and backend systems, allowing users to upload images for classification and receive predictions via a user-friendly interface on both web and mobile platforms.
```py
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
```

# Tech Stack Documentation
**Client Side**: [React Native](https://reactnative.dev/), [Tailwind](https://tailwindcss.com/), 
**Serverless**: [GCP](https://cloud.google.com/functions), [Tensorflow](https://www.tensorflow.org/), [FastAPI](https://fastapi.tiangolo.com/)

# App preview

**Web Application Link:** [Tomato Web App](https://dr-tomato.uc.r.appspot.com/)
**Android Application:** [Tomato Android App](https://drive.google.com/file/d/1GJlmTgCxODij2xQah5_6fyuDYd3xWdmB/view)

