import gradio as gr
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import load_digits

# -----------------------
# Load trained model
# -----------------------
clf = joblib.load("Digit-classification.pkl")

# Load sklearn digits to get feature names
digits = load_digits()
feature_names = digits.feature_names

# Define your top 36 features (the same used for training)
# Make sure this matches your notebook
top_feature = [
    'pixel_4_6', 'pixel_2_1', 'pixel_3_4', 'pixel_3_1', 'pixel_2_4',
    'pixel_7_6', 'pixel_1_3', 'pixel_3_2', 'pixel_5_5', 'pixel_3_5',
    'pixel_2_2', 'pixel_4_5', 'pixel_5_2', 'pixel_6_7', 'pixel_6_5',
    'pixel_2_6', 'pixel_1_5', 'pixel_4_4', 'pixel_7_7', 'pixel_2_3',
    'pixel_0_7', 'pixel_6_2', 'pixel_2_5', 'pixel_5_3', 'pixel_3_3',
    'pixel_1_4', 'pixel_5_4', 'pixel_7_1', 'pixel_0_1', 'pixel_2_7',
    'pixel_4_3', 'pixel_6_1', 'pixel_3_6', 'pixel_6_3', 'pixel_4_1',
    'pixel_6_4'
]

# -----------------------
# Prediction function
# -----------------------
def predict_digit(image):
    # Convert uploaded image to 8x8 grayscale
    img = image.convert("L").resize((8, 8))
    arr = np.array(img)

    # Scale 0–255 → 0–16 like sklearn digits
    arr = (arr / 16).astype(int)

    # Flatten to 64 pixels
    arr_flat = arr.flatten()

    # Create DataFrame with same feature names
    df_img = pd.DataFrame([arr_flat], columns=feature_names)

    # Select only the top 36 features
    df_selected = df_img[top_feature]

    # Predict
    pred = clf.predict(df_selected)[0]
    return int(pred)

interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil", label="Upload Digit Image"),
    outputs=gr.Label(label="Predicted Digit"),
    title="🖼️ Digit Classifier",
    description="Upload an 8x8 digit image (similar to sklearn digits dataset)."
)

if __name__ == "__main__":
    interface.launch() 