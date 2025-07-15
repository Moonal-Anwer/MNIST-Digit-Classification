import gradio as gr
import numpy as np
from PIL import Image, ImageOps
# import joblib # No longer needed for loading .h5 Keras models
import tensorflow as tf # Import TensorFlow

# Load the trained model
# model = joblib.load("mlp_model.h5") # THIS IS THE INCORRECT LINE
model = tf.keras.models.load_model("mlp_model.h5") # THIS IS THE CORRECT LINE

# Preprocess function
def preprocess(image_input):
    # Gradio's Sketchpad might return a dictionary with 'image' and 'mask' keys,
    # or just the image array if it's a simpler setup.
    # We need to ensure we're getting the actual numpy array for the image.

    image_array = None

    if isinstance(image_input, dict):
        # In newer Gradio versions or with specific Sketchpad configurations,
        # the 'image' key might hold the composite image (all layers merged).
        if 'image' in image_input and image_input['image'] is not None:
            image_array = image_input['image']
        # If 'image' is not present or None, but 'composite' is, use that.
        # This handles cases where sketchpad might return 'composite' instead of 'image'
        elif 'composite' in image_input and image_input['composite'] is not None:
            image_array = image_input['composite']
        # If no explicit image or composite, try to find a numpy array directly
        # within the dictionary's values if there's only one. This is a less
        # common fallback but can cover edge cases.
        else:
            for value in image_input.values():
                if isinstance(value, np.ndarray):
                    image_array = value
                    break
    elif isinstance(image_input, np.ndarray):
        # If image_input is already a numpy array (older Gradio or specific settings)
        image_array = image_input

    # If after all checks, image_array is still None, it means no valid drawing was provided.
    if image_array is None:
        # Return a black 28x28 image array as a default empty input
        return np.zeros((1, 784))

    # Ensure the image_array is of integer type before converting to PIL Image
    # Gradio might return float arrays from sketchpad depending on its internal processing.
    if image_array.dtype != np.uint8:
        # Scale and convert to uint8, assuming values are between 0-255 or 0-1
        if image_array.max() > 1.0: # If values are 0-255 already or higher
            image_array = image_array.astype(np.uint8)
        else: # If values are 0-1 (normalized floats)
            image_array = (image_array * 255).astype(np.uint8)


    img = Image.fromarray(image_array).convert("L")  # Convert to grayscale
    img = ImageOps.invert(img)                        # Invert to white background (MNIST has white digits on black)
    img = img.resize((28, 28))                        # Resize to 28x28 (MNIST standard)
    img_array = np.array(img) / 255.0                 # Normalize pixel values to 0-1
    flat = img_array.flatten().reshape(1, -1)         # Flatten to shape (1, 784) for model input
    return flat

# Prediction function
def predict_digit(image_input):
    processed = preprocess(image_input)
    # Check if the processed array is all zeros (indicating an empty sketchpad)
    if np.all(processed == 0):
        return "Please draw a digit." # Provide a user-friendly message for empty input
    else:
        # For Keras models, use model.predict, which returns probabilities.
        # We then use argmax to get the predicted digit.
        prediction_probabilities = model.predict(processed)
        predicted_digit = np.argmax(prediction_probabilities)
        return f"Predicted Digit: {int(predicted_digit)}"

# Gradio interface
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="Draw a digit (0–9)"),
    outputs="text",
    title="MLP Digit Recognizer", # Changed title to reflect MLP
    theme="soft"
)

# Launch app
if __name__ == "__main__":
    interface.launch()