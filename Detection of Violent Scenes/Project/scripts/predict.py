import cv2
import numpy as np
from keras.models import load_model
import os

def predict_image(model, image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    # Resize the image to 224x224 (for the model's expected input size)
    image = cv2.resize(image, (224, 224))

    # Convert the image to a 4D tensor for prediction
    image = np.expand_dims(image, axis=0)

    # Normalize the image (if required by the model)
    image = image / 255.0

    # Predict using the model
    predictions = model.predict(image)

    # Return the predicted class (0: non-violent, 1: violent)
    return np.argmax(predictions, axis=1)

if __name__ == "__main__":
    # Define the path to the saved model and the test image
    model_path = 'D:/Projects/Detection of Violent Scenes/models/violence_detection_model.h5'
    test_image_path = 'D:/Projects/Detection of Violent Scenes/test_image.jpg'  # Replace with your test image path

    # Ensure the model file exists before loading
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
    else:
        # Load the trained model
        model = load_model(model_path)

        # Predict the class of the test image
        result = predict_image(model, test_image_path)

        if result is not None:
            if result == 0:
                print("Prediction: Non-Violent")
            else:
                print("Prediction: Violent")
