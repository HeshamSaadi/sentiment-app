
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import requests
import zipfile
import os

# GitHub repository details
GITHUB_REPO_OWNER = "HeshamSaadi"
GITHUB_REPO_NAME = "sentiment-app"
RELEASE_ASSET_NAME = "models.zip"

# Function to download and extract model from GitHub release
@st.cache_resource
def load_model_from_github():
    st.write("Downloading model from GitHub...")
    try:
        # Get the latest release information
        releases_url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases/latest"
        response = requests.get(releases_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        release_info = response.json()

        asset_url = None
        for asset in release_info["assets"]:
            if asset["name"] == RELEASE_ASSET_NAME:
                asset_url = asset["url"]
                break

        if not asset_url:
            st.error(f"Release asset \'{RELEASE_ASSET_NAME}\' not found.")
            return None, None, None

        # Download the asset
        headers = {"Accept": "application/octet-stream"}
        asset_response = requests.get(asset_url, headers=headers, stream=True)
        asset_response.raise_for_status()

        zip_path = "models.zip"
        with open(zip_path, "wb") as f:
            for chunk in asset_response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")

        # Load the model, tokenizer, and label mapping
        model_path = os.path.join("models", "simplified_lstm_20250612-172438_best.h5")
        tokenizer_path = os.path.join("models", "simplified_lstm_20250612-172438_tokenizer.pickle")
        label_mapping_path = os.path.join("models", "simplified_lstm_20250612-172438_label_mapping.pickle")

        with open(label_mapping_path, "rb") as handle:
            label_mapping = pickle.load(handle)

        # Register 'NotEqual' as a custom object if it's a known TensorFlow operation
        # This is a common issue when loading models saved with newer TF versions or specific ops
        custom_objects = {"NotEqual": tf.math.not_equal}
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
        with open(label_mapping_path, "rb") as handle:
            label_mapping = pickle.load(handle)

        st.success("Model loaded successfully!")
        return model, tokenizer, label_mapping

    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model from GitHub: {e}")
        return None, None, None
    except zipfile.BadZipFile:
        st.error("Downloaded file is not a valid zip file.")
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, None, None

model, tokenizer, label_mapping = load_model_from_github()

if model and tokenizer and label_mapping:
    # Streamlit UI
    st.title("Sentiment Analysis App")

    user_input = st.text_area("Enter your text here:", "")

    if st.button("Analyze Sentiment"):
        if user_input:
            # Preprocess the input text
            sequence = tokenizer.texts_to_sequences([user_input])
            padded_sequence = pad_sequences(sequence, maxlen=model.input_shape[1])

            # Predict sentiment
            prediction = model.predict(padded_sequence)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Map predicted class to label
            # Assuming label_mapping is a dictionary where keys are class indices and values are labels
            sentiment_label = "Unknown"
            for label, index in label_mapping.items():
                if index == predicted_class:
                    sentiment_label = label
                    break

            st.write(f"Sentiment: **{sentiment_label}**")
        else:
            st.warning("Please enter some text to analyze.")
else:
    st.error("Could not load the model. Please check the GitHub repository and asset name.")


