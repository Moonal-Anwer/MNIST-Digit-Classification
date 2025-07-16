import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import joblib
import xgboost as xgb # Import xgboost

# --- Model Loading ---
try:
    # Ensure this path is correct for your xgb_model.pkl
    model = joblib.load("xgb_model.pkl")
    print("XGBoost model loaded successfully.")
except Exception as e:
    print(f"Error loading XGBoost model: {e}")
    print("Please ensure 'xgb_model.pkl' is in the correct directory.")
    # Fallback to a dummy model is not ideal for a real application,
    # but for demonstration, we'll keep a placeholder message.
    # In a production environment, you'd likely raise an error or exit.
    model = None # Set model to None to indicate failure to load
    print("Failed to load the actual model. Prediction will not work correctly.")


def predict_digit(img) -> tuple[dict, str]:
    """
    Preprocesses the input image and predicts the digit using the loaded XGBoost model.

    Args:
        img: The input from the Gradio Sketchpad.

    Returns:
        tuple[dict, str]: A tuple containing:
            - A dictionary of "pseudo" confidence scores (based on direct prediction if not probabilistic).
            - A string indicating the predicted digit.
    """
    if model is None:
        return {str(i): 0.0 for i in range(10)}, "Model not loaded. Cannot predict."

    if img is None:
        return {str(i): 0.0 for i in range(10)}, "No digit drawn."

    # Handle cases where Sketchpad returns a dictionary (e.g., {'background': ..., 'composite': ...})
    if isinstance(img, dict):
        if 'composite' in img and img['composite'] is not None:
            img_array = img['composite']
        else:
            return {str(i): 0.0 for i in range(10)}, "No valid image data found in Sketchpad output."
    elif isinstance(img, np.ndarray):
        img_array = img
    else:
        return {str(i): 0.0 for i in range(10)}, "Unexpected input type from Sketchpad."

    # Convert NumPy array to PIL Image (assuming RGB input from Sketchpad)
    img_pil = Image.fromarray(img_array.astype('uint8'), 'RGB')

    # Convert to grayscale
    img_pil = img_pil.convert('L') # 'L' mode for grayscale

    # Resize to 28x28 pixels (standard for MNIST-like datasets)
    img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS) # Use a high-quality resampling filter

    # Invert colors: Gradio Sketchpad typically draws black on white.
    # MNIST-like models expect white digit on black background.
    img_pil = ImageOps.invert(img_pil)

    # Convert to numpy array and normalize to 0-1 range
    # Ensure the image is flattened for XGBoost
    img_array_processed = np.array(img_pil) / 255.0
    img_array_processed = img_array_processed.flatten().reshape(1, -1) # Flatten to (1, 784)

    # Make prediction using the XGBoost model
    # XGBoost's predict typically returns the class label directly for classification
    # or raw scores. If it's a multi-class classifier, it often returns the predicted class index.
    # If you trained with `objective='multi:softprob'`, `predict_proba` (or simply `predict`)
    # would give probabilities. Assuming `predict` gives the direct class prediction.
    predicted_digit_index = model.predict(img_array_processed)[0] # Get the first (and only) prediction

    # For XGBoost, especially with 'multi:softmax' objective, `predict` returns the class directly.
    # If your model was trained with 'multi:softprob', you would use `model.predict_proba()`
    # to get probabilities for each class.
    # Since we're dealing with `predict`, we'll create a "confidence" dictionary
    # where the predicted digit has a confidence of 1.0 and others 0.0,
    # unless your XGBoost model's `predict` method actually gives a probability-like output
    # (which is less common for `predict` on a softmax objective).

    # If your model was trained with `objective='multi:softprob'`, you could do:
    # probabilities = model.predict_proba(img_array_processed)[0]
    # confidences = {str(i): float(probabilities[i]) for i in range(10)}

    # Assuming `model.predict` directly gives the predicted class label (0-9)
    # We'll create a placeholder for confidences as XGBoost `predict` doesn't directly
    # give a probability distribution like `softmax`.
    confidences = {str(i): 0.0 for i in range(10)}
    # Assign 1.0 to the predicted digit for demonstration of "confidence"
    confidences[str(int(predicted_digit_index))] = 1.0

    predicted_digit = str(int(predicted_digit_index)) # Ensure it's an integer and then string

    return confidences, f"Predicted Digit: {predicted_digit}"

# Create the Gradio interface
# The Sketchpad allows users to draw directly.
# The output will be a Label showing probabilities and a Textbox for the predicted digit.
iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(
        label="Draw a digit (0-9) here",
        image_mode="RGB", # Sketchpad outputs RGB, we convert to L (grayscale) inside the function
        # Removed 'brush_color' and 'brush_radius' as they are not supported
        # by gr.Sketchpad in some Gradio versions. Default brush settings will be used.
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