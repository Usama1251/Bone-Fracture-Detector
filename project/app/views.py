import os
from django.shortcuts import render, HttpResponse
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Define the path to your trained model
model_path = r'C:\Usama\Semester\CV\project\model.keras'  # Using raw string

# Load the trained model for predictions
loaded_model = load_model(model_path)

def index(request):
    result = None

    if request.method == 'POST' and 'image' in request.FILES:
        # Get the uploaded image from the form
        uploaded_image = request.FILES['image']

        # Save the uploaded image to a temporary file
        temp_image_path = os.path.join(r'C:\Usama\Semester\CV\project\tmp', uploaded_image.name)
        with open(temp_image_path, 'wb') as temp_image:
            for chunk in uploaded_image.chunks():
                temp_image.write(chunk)

        # Load and preprocess the new image
        img_size = (224, 224)
        new_image = load_img(temp_image_path, target_size=img_size)
        new_image_array = img_to_array(new_image)
        new_image_array = np.expand_dims(new_image_array, axis=0) / 255.0  # Normalize the image

        # Make a prediction using the loaded model
        prediction = loaded_model.predict(new_image_array)

        # Apply a threshold (commonly 0.5 for binary classification)
        threshold = 0.5
        predicted_class = 1 if prediction >= threshold else 0

        # Display the result on the webpage
        if predicted_class == 1:
            result = "The bone is cracked."
        else:
            result = "The bone is not cracked."

        # Remove the temporary image file
        os.remove(temp_image_path)

    return render(request, "web.html", {'result': result})
