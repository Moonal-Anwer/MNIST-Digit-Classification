import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import joblib


model = joblib.load("xgb_model.pkl")

# Preprocess function
def preprocess(image_input):

    image_array = None

    if isinstance(image_input, dict):
        
        if 'image' in image_input and image_input['image'] is not None:
            image_array = image_input['image']
        
        elif 'composite' in image_input and image_input['composite'] is not None:
            image_array = image_input['composite']
        
        else:
            for value in image_input.values():
                if isinstance(value, np.ndarray):
                    image_array = value
                    break
    elif isinstance(image_input, np.ndarray):
        
        image_array = image_input
    if image_array is None:
        return np.zeros((1, 784))
    if image_array.dtype != np.uint8:
        if image_array.max() > 1.0: 
            image_array = image_array.astype(np.uint8)
        else: 
            image_array = (image_array * 255).astype(np.uint8)


    img = Image.fromarray(image_array).convert("L")  
    img = ImageOps.invert(img)                       
    img = img.resize((28, 28))                        
    img_array = np.array(img) / 255.0                 
    flat = img_array.flatten().reshape(1, -1)         
    return flat


def predict_digit(image_input):
    processed = preprocess(image_input)
    
    if np.all(processed == 0):
        return "Please draw a digit."
    else:
        prediction = model.predict(processed)[0]
        return f"Predicted Digit: {int(prediction)}"

interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="Draw a digit (0–9)"),
    outputs="text",
    title="XGBoost Digit Recognizer",
    theme="soft"
)

# Launch app
if __name__ == "__main__":
    interface.launch()