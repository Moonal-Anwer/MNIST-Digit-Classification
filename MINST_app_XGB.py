import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import joblib
import xgboost as xgb

model = joblib.load("xgb_model.pkl")

def predict_digit(img) -> str:
    if model is None:
        return "Model not loaded. Cannot predict."

    if img is None:
        return "No digit drawn."

    if isinstance(img, dict):
        if 'composite' in img and img['composite'] is not None:
            img_array = img['composite']
        else:
            return "No valid image data found in Sketchpad output."
    elif isinstance(img, np.ndarray):
        img_array = img
    else:
        return "Unexpected input type from Sketchpad."

    img_pil = Image.fromarray(img_array.astype('uint8'), 'RGB')
    img_pil = img_pil.convert('L')
    img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
    img_pil = ImageOps.invert(img_pil)

    img_array_processed = np.array(img_pil) / 255.0
    img_array_processed = img_array_processed.flatten().reshape(1, -1)

    predicted_digit_index = model.predict(img_array_processed)[0]
    predicted_digit = str(int(predicted_digit_index))
    return f"Predicted Digit: {predicted_digit}"

css_code = """
body, .gradio-container {
    background-color: #fce4ec !important;
}

.gr-button {
    background-color: #ec407a !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: bold;
}
"""

custom_theme = gr.themes.Base(
    primary_hue="pink",
    secondary_hue="purple",
    neutral_hue="rose",
    font=["Comic Sans MS", "cursive"]
)

iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(
        label="Draw a digit (0-9) here",
        image_mode="RGB",
    ),
    outputs=gr.Textbox(label="Prediction"),
    title="Handwritten Digit Recognizer",
    description="Draw a single digit (0-9) in the sketchpad, and the model will predict what it is!",
    allow_flagging="never",
    theme=custom_theme,
    css=css_code
)

iface.launch()


