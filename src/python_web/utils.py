import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

CLASS_NAMES = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

IMG_SIZE_VGG = 224
IMG_SIZE_CNN = 28

def load_history(path: str):
    with open(path, "r") as f:
        return json.load(f)

def plot_history_dict(hist: dict, title_prefix="Model"):
    # Loss
    fig1 = plt.figure(figsize=(6, 4))
    if "loss" in hist: plt.plot(hist["loss"], label="train")
    if "val_loss" in hist: plt.plot(hist["val_loss"], label="val")
    plt.title(f"{title_prefix}: Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(True); plt.legend()
    # Accuracy
    fig2 = plt.figure(figsize=(6, 4))
    if "accuracy" in hist: plt.plot(hist["accuracy"], label="train")
    if "val_accuracy" in hist: plt.plot(hist["val_accuracy"], label="val")
    plt.title(f"{title_prefix}: Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.grid(True); plt.legend()
    return fig1, fig2

def pil_to_gray(pil_img: Image.Image) -> Image.Image:
    return pil_img.convert("L")

def preprocess_for_cnn(pil_img: Image.Image) -> np.ndarray:
    img = pil_to_gray(pil_img).resize((IMG_SIZE_CNN, IMG_SIZE_CNN))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)  # (28,28,1)
    arr = np.expand_dims(arr, axis=0)   # (1,28,28,1)
    return arr

def preprocess_for_vgg(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((IMG_SIZE_VGG, IMG_SIZE_VGG))
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)   # (1,224,224,3)
    arr = preprocess_input(arr)
    return arr

def predict_probs(model, x: np.ndarray):
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return probs, pred_idx

def probs_table(probs: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"class": CLASS_NAMES, "probability": probs})
    return df.sort_values("probability", ascending=False).reset_index(drop=True)

def plot_probs_bar(df: pd.DataFrame, title="Class probabilities"):
    fig = plt.figure(figsize=(7,4))
    plt.bar(df["class"], df["probability"])
    plt.xticks(rotation=45, ha="right")
    plt.ylim([0, 1.0])
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    return fig
