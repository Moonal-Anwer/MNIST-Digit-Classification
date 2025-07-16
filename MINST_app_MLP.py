import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf # Import TensorFlow

model = tf.keras.models.load_model("mlp_model.h5")



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
    title="Handwritten Digit Recognizer",
    description="Draw a single digit (0-9) in the sketchpad, and the model will predict what it is!",
    allow_flagging="never" 
)


iface.launch()
