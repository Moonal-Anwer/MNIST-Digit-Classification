import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf # Import TensorFlow

# Load the pre-trained model
# IMPORTANT: Ensure 'mlp_model.h5' is in the same directory as this script,
# or provide the full path to the model file.
try:
    model = tf.keras.models.load_model("mlp_model.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'mlp_model.h5' is in the correct directory.")
    # Create a dummy model for demonstration if the actual model fails to load
    # In a real scenario, you would handle this more robustly, e.g., by exiting or providing a fallback.
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    print("Using a dummy model for demonstration purposes.")


def predict_digit(img) -> tuple[dict, str]: # Removed type hint for now to handle dict or array
    """
    Preprocesses the input image and predicts the digit using the loaded model.

    Args:
        img: The input from the Gradio Sketchpad, which could be a NumPy array or a dictionary.

    Returns:
        tuple[dict, str]: A tuple containing:
            - A dictionary of confidence scores for each digit (0-9).
            - A string indicating the predicted digit.
    """
    if img is None:
        return {str(i): 0.0 for i in range(10)}, "No digit drawn."

    # Handle cases where Sketchpad returns a dictionary (e.g., {'background': ..., 'composite': ...})
    # The actual drawn image is usually in the 'composite' key for Sketchpad
    if isinstance(img, dict):
        if 'composite' in img and img['composite'] is not None:
            img_array = img['composite']
        else:
            # If 'composite' is not available or is None, it might be an empty sketchpad
            # or a different structure. Return a message indicating no valid image data.
            return {str(i): 0.0 for i in range(10)}, "No valid image data found in Sketchpad output."
    elif isinstance(img, np.ndarray):
        img_array = img
    else:
        # Unexpected input type
        return {str(i): 0.0 for i in range(10)}, "Unexpected input type from Sketchpad."

    # Now img_array should be a NumPy array, proceed with processing
    # Convert NumPy array to PIL Image
    # Gradio Sketchpad with image_mode="RGB" returns a NumPy array (H, W, 3)
    img_pil = Image.fromarray(img_array.astype('uint8'), 'RGB')

    # Convert to grayscale
    img_pil = img_pil.convert('L')

    # Resize to 28x28 pixels
    img_pil = img_pil.resize((28, 28))

    # Invert colors: Gradio Sketchpad typically draws black on white.
    # MNIST models expect white digit on black background.
    img_pil = ImageOps.invert(img_pil)

    # Convert to numpy array and normalize to 0-1 range
    img_array_processed = np.array(img_pil) / 255.0

    # Reshape for the model: (batch_size, height, width, channels)
    # This was (1, 28, 28, 1) which is not compatible with a flattened input model
    # Flatten the image to (1, 784) for the MLP model
    img_array_processed = img_array_processed.reshape(1, 28 * 28) # Flatten to (1, 784)

    # Make prediction
    predictions = model.predict(img_array_processed)[0] # Get the first (and only) prediction

    # Get confidence scores
    confidences = {str(i): float(predictions[i]) for i in range(10)}

    # Get the predicted digit
    predicted_digit = str(np.argmax(predictions))

    return confidences, f"Predicted Digit: {predicted_digit}"

# Create the Gradio interface
# The Sketchpad allows users to draw directly.
# The output will be a Label showing probabilities and a Textbox for the predicted digit.
iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(
        label="Draw a digit (0-9) here",
        image_mode="RGB", # Sketchpad outputs RGB, we convert to L (grayscale) inside the function
        # Removed 'shape' argument as it's not supported by gr.Sketchpad in recent Gradio versions.
        # Removed 'brush_color' and 'brush_radius' arguments as they are also not supported
        # in some Gradio versions. Default brush settings will be used.
        # Removed 'invert_colors' argument as it's not supported in some Gradio versions.
    ),
    outputs=[
        gr.Label(label="Confidence Scores", num_top_classes=10),
        gr.Textbox(label="Prediction")
    ],
    title="Handwritten Digit Recognizer",
    description="Draw a single digit (0-9) in the sketchpad, and the model will predict what it is!",
    allow_flagging="never" # Disable flagging feature
)

# Launch the Gradio interface
iface.launch()
