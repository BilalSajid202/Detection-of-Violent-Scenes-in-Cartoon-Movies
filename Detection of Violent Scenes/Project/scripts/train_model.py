import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Ensure the necessary directories exist
model_save_path = 'D:/Projects/Detection of Violent Scenes/models/violence_detection_model.h5'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Define your CNN model with additional Dropout for regularization
def create_model():
    model = Sequential()

    # Convolutional layers with dropout to prevent overfitting
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Dropout layer

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Dropout layer

    model.add(Flatten())

    # Dense layers with dropout for further regularization
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout layer
    model.add(Dense(2, activation='softmax'))  # 2 classes: violent, non-violent

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the data directories
train_data_dir = 'D:/Projects/Detection of Violent Scenes/extracted_frames'
batch_size = 32

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,  # 20% of data for validation
    rotation_range=15,  # Rotation for augmentation
    width_shift_range=0.1,  # Horizontal shift for augmentation
    height_shift_range=0.1,  # Vertical shift for augmentation
    shear_range=0.1,  # Shearing for augmentation
    zoom_range=0.1,  # Zoom for augmentation
    horizontal_flip=True, 
    fill_mode='nearest'
)

# Train data generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Create and compile the model
model = create_model()

# ModelCheckpoint to save the best model
checkpoint = ModelCheckpoint(
    model_save_path,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# Train the model with validation and checkpointing
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

print(f"Model saved at {model_save_path}")
