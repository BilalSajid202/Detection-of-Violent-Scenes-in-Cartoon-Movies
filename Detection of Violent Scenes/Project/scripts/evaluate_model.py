import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Paths
model_path = 'D:/Projects/Detection of Violent Scenes/models/violence_detection_model.h5'
validation_data_dir = 'D:/Projects/Detection of Violent Scenes/extracted_frames'

# Load the trained model
model = load_model(model_path)

# Set up data generator for validation
datagen = ImageDataGenerator(rescale=1.0/255.0)
validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Check if validation generator has images
if validation_generator.samples == 0:
    print("No images found in the validation directory. Please check the path or add images.")
else:
    # Get true labels and predictions
    y_true = validation_generator.classes  # Ground truth labels
    y_pred = np.argmax(model.predict(validation_generator), axis=1)  # Model predictions

    # Calculate accuracy and display results
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Non-Violent", "Violent"])

    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", report)
