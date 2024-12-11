import tensorflow as tf
import numpy as np
import cv2
import os

def load_model(model_path):
    """
    Load the pre-trained .keras model from the given path.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess the image to match the input requirements of the model.
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        exit(1)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Unable to read image: {image_path}")
        exit(1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)  # Resize to model's input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict(model, image):
    """
    Predict whether the image is pneumonia or normal.
    """
    predictions = model.predict(image)
    probability = predictions[0][0]
    if probability > 0.5:
        return "Pneumonia", probability
    else:
        return "Normal", 1 - probability

# --- Main Program ---
if __name__ == "__main__":
    #Load model
    model_path = "model/model.h5"
    #Load image
    # test_image_path = "test/Normal6.jpeg" 
    #Process Model
    model = load_model(model_path)
    #Preocssed image
    # processed_image = preprocess_image(test_image_path)
    
    # # Make a prediction
    # label, confidence = predict(model, processed_image)
    # print(f"Prediction: {label} with confidence {confidence:.2f}")


    #Test all picture
    test_folder = "test"
    for image_name in os.listdir(test_folder):
        test_image_path = os.path.join(test_folder, image_name)
        processed_image = preprocess_image(test_image_path)
        label, confidence = predict(model, processed_image)
        print(f"File: {image_name} => Prediction: {label} with confidence {confidence:.2f}")
