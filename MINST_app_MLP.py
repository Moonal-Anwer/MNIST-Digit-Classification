import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

model = tf.keras.models.load_model("mlp_model_final.h5")

def predict_digit(img) -> tuple[dict, str]:
    if img is None:
        return {str(i): 0.0 for i in range(10)}, "No digit drawn."

    if isinstance(img, dict):
        if 'composite' in img and img['composite'] is not None:
            img_array = img['composite']
        else:
            return {str(i): 0.0 for i in range(10)}, "No valid image data found in Sketchpad output."
    elif isinstance(img, np.ndarray):
        img_array = img
    else:
        return {str(i): 0.0 for i in range(10)}, "Unexpected input type from Sketchpad."

    img_pil = Image.fromarray(img_array.astype('uint8'), 'RGB')
    img_pil = img_pil.convert('L')
    img_pil = img_pil.resize((28, 28))
    img_pil = ImageOps.invert(img_pil)

    img_array_processed = np.array(img_pil) / 255.0

    img_array_processed = img_array_processed.reshape(1, 28 * 28)

    predictions = model.predict(img_array_processed)[0]

    confidences = {str(i): float(predictions[i]) for i in range(10)}
    predicted_digit = str(np.argmax(predictions))

    return confidences, f"Predicted Digit: {predicted_digit}"

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
.gradio-container {
    border-radius: 15px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
.gr-label, .gr-textbox {
    border-radius: 8px;
}
"""

custom_theme = gr.themes.Base(
    primary_hue="pink",
    secondary_hue="purple",
    neutral_hue="rose",
    font=["Comic Sans MS", "cursive", "sans-serif"]
).set(
    button_primary_background_fill="#ec407a",
    button_primary_background_fill_hover="#d81b60",
)

iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(
        label="Draw a digit (0-9) here",
        image_mode="RGB",
    ),
    outputs=[
        gr.Label(label="Confidence Scores", num_top_classes=10),
        gr.Textbox(label="Prediction")
    ],
    title=" Handwritten Digit Recognizer",
    description="Draw a single digit (0-9) in the sketchpad, and the model will predict what it is! Try to draw clearly for best results.",
    allow_flagging="never",
    theme=custom_theme,
    css=css_code
)

if __name__ == "__main__":
    iface.launch()
