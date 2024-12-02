import os
import cv2
import numpy as np

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            # Resize images to 224x224
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

def load_dataset():
    violent_folder = 'D:/Projects/Detection of Violent Scenes/extracted_frames/violent'
    non_violent_folder = 'D:/Projects/Detection of Violent Scenes/extracted_frames/non_violent'

    # Load violent images (label 1)
    violent_images, violent_labels = load_images_from_folder(violent_folder, 1)

    # Load non-violent images (label 0)
    non_violent_images, non_violent_labels = load_images_from_folder(non_violent_folder, 0)

    # Combine datasets
    X = np.concatenate((violent_images, non_violent_images), axis=0)
    y = np.concatenate((violent_labels, non_violent_labels), axis=0)

    return X, y

if __name__ == "__main__":
    X, y = load_dataset()
    print(f"Dataset loaded with {len(X)} images")
 