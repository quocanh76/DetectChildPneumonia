import tensorflow as tf
import numpy as np
import cv2
import os
from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk

# Load the model
model = None

def load_model(model_path):
    global model
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def preprocess_image(image_path, target_size=(224, 224)):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image_path):
    image = preprocess_image(image_path)
    if image is not None and model is not None:
        prediction = model.predict(image)
        return prediction
    return None

def browse_image():
    filename = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if filename:
        display_image(filename)
        prediction = predict(filename)
        if prediction is not None:
            messagebox.showinfo("Prediction", f"Prediction: {'Normal' if prediction < 0.5 else 'Pneumonia'}")

def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((300, 300), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.grid(column=0, row=1, columnspan=2, pady=20)

# Initialize the GUI
root = Tk()
root.title("Image Prediction App")
root.geometry("400x400")

load_model("model\model.h5")

browse_button = Button(root, text="Browse Image", command=browse_image)
browse_button.grid(column=0, row=0, columnspan=2, pady=20)

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)

root.mainloop()