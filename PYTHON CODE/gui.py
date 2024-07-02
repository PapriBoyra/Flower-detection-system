import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array

# Define the main GUI application
class Flower:
    def __init__(self, root):
        self.root = root
        self.root.title("Flower Detection")
        
        # Load trained model 
        self.model = tf.keras.models.load_model('J:/PROJECT/Final/archive/flowers/flowers.keras')
        
        # Create GUI 
        self.create_gui()

    def create_gui(self):
        # Create a frame for the image canvas and prediction label
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Create the image label
        self.image_label = tk.Label(frame)
        self.image_label.pack(pady=7)

        # Create a button frame
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=3)

        # Load Model Button
        self.load_model_button = tk.Button(button_frame, text="Load Model", command=self.load_model, padx=5, pady=3, bg='purple', fg='white', font=('Times New Roman', 10))
        self.load_model_button.grid(row=0, column=0, padx=5)

        # Predict Button
        self.predict_button = tk.Button(button_frame, text="Predict", command=self.predict, padx=5, pady=3, bg='Royal blue', fg='white', font=('Times New Roman', 10))
        self.predict_button.grid(row=0, column=1, padx=5)

        # Display image details
        self.image_details_label = tk.Label(frame, text="", font=('Times New Roman', 10))
        self.image_details_label.pack(pady=10)

        # Create a status label
        self.status_label = tk.Label(frame, text="", fg="blue", font=('Times New Roman', 10))
        self.status_label.pack(pady=10)

    def load_model(self):
        # Load your trained model here
        self.model = tf.keras.models.load_model('J:/PROJECT/Final/archive/flowers/flowers.keras')
        self.status_label.config(text="Model Loaded")

    def predict(self):
        # Load and preprocess image for prediction
        image_path = filedialog.askopenfilename(title="Select an image for prediction", filetypes=[("Image Files", "*.jpg *.png")])
        if image_path:
            # Load the image
            image = load_img(image_path, target_size=(224, 224))
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = image_array / 255.0

            # Make a prediction using the loaded model
            predictions = self.model.predict(image_array)
            predicted_class = np.argmax(predictions)
            accuracy_score = predictions[0][predicted_class] * 100  # Convert to percentage

            # Display the prediction
            class_names = ["daisy 0", "dandelion 1", "rose 2", "sunflower 3", "tulip 4"]
            prediction_text = f"Predicted: {class_names[predicted_class]} (Accuracy: {accuracy_score:.2f}%)"
            self.status_label.config(text=prediction_text)

            # Display the selected image
            img = Image.open(image_path)
            img.thumbnail((400, 400))  # Resize large images to fit the window
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            # Display image details
            image_details_text = f"Image Details:\nDimensions: {image_array.shape}\nResolution: {img.size}"
            self.image_details_label.config(text=image_details_text)

# Create the main GUI 
root = tk.Tk()
app = Flower(root)
root.mainloop()
