import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import zipfile
import os
import pickle
import tensorflow_addons as tfa

# GitHub repository details
GITHUB_REPO_OWNER = "HeshamSaadi"
GITHUB_REPO_NAME = "sentiment-app"
GITHUB_RELEASE_TAG = "Model-Release" # Assuming this is the tag for your release
GITHUB_MODEL_ZIP_NAME = "models.zip"

# Paths for model and tokenizer within the extracted zip
MODEL_FILE_NAME = "simplified_lstm_20250612-172438_best.h5"
TOKENIZER_FILE_NAME = "simplified_lstm_20250612-172438_tokenizer.pickle"
LABEL_MAPPING_FILE_NAME = "simplified_lstm_20250612-172438_label_mapping.pickle"

# Custom FocalLoss class from the notebook
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name="focal_loss", **kwargs):
        super(FocalLoss, self).__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -tf.keras.backend.mean(self.alpha * tf.keras.backend.pow(1. - pt, self.gamma) * tf.keras.backend.log(pt), axis=-1)
        return loss

    def get_config(self):
        config = super(FocalLoss, self).get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
        })
        return config

@st.cache_resource
def load_model_and_tokenizer():
    model = None
    tokenizer = None
    label_mapping = None

    st.write("Downloading model from GitHub...")
    try:
        # Construct the URL for the release asset
        release_url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases/tags/{GITHUB_RELEASE_TAG}"
        headers = {"Accept": "application/vnd.github.v3+json"}
        response = requests.get(release_url, headers=headers)
        response.raise_for_status() # Raise an exception for HTTP errors
        release_info = response.json()

        asset_url = None
        for asset in release_info["assets"]:
            if asset["name"] == GITHUB_MODEL_ZIP_NAME:
                asset_url = asset["url"]
                break

        if not asset_url:
            st.error(f"Could not find asset \'{GITHUB_MODEL_ZIP_NAME}\' in release \'{GITHUB_RELEASE_TAG}\\\'. Please check the release name and asset name.")
            return None, None, None

        # Download the zip file
        st.write(f"Found asset: {GITHUB_MODEL_ZIP_NAME}. Downloading...")
        asset_response = requests.get(asset_url, headers={
            "Accept": "application/octet-stream",
            "Authorization": f"token {os.environ.get("GITHUB_TOKEN")}" # Use a token if repo is private or for higher rate limits
        }, stream=True)
        asset_response.raise_for_status()

        zip_path = GITHUB_MODEL_ZIP_NAME
        with open(zip_path, "wb") as f:
            for chunk in asset_response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.write("Download complete. Extracting files...")

        # Extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Extract all contents directly into the current directory
            # This assumes the zip contains a \'models\' directory at its root
            zip_ref.extractall(".")
        st.write("Extraction complete.")

        # Define paths to the extracted files
        # Assuming the zip extracts to a \'models\' directory at the root
        model_path = os.path.join("models", MODEL_FILE_NAME)
        tokenizer_path = os.path.join("models", TOKENIZER_FILE_NAME)
        label_mapping_path = os.path.join("models", LABEL_MAPPING_FILE_NAME)

        # Custom objects dictionary for model loading
        custom_objects = {
            "FocalLoss": FocalLoss, # Registering the custom FocalLoss class
            # Add any other custom objects from tensorflow_addons if they are explicitly used
            # For example, if a custom loss function from TFA was used:
            # \'Addons>SigmoidFocalCrossEntropy\': tfa.losses.SigmoidFocalCrossEntropy
            # You might need to inspect your model\'s config if this doesn\'t work.
        }

        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path)

        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
        with open(label_mapping_path, "rb") as handle:
            label_mapping = pickle.load(handle)

        st.success("Model and tokenizer loaded successfully!")

    except requests.exceptions.RequestException as e:
        st.error(f"Network error during model download: {e}")
    except zipfile.BadZipFile:
        st.error("Downloaded file is not a valid zip file.")
    except FileNotFoundError as e:
        st.error(f"File not found after extraction: {e}. Please check the paths within your zip file.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.error("Could not load the model. Please check the GitHub repository and asset name.")

    return model, tokenizer, label_mapping

model, tokenizer, label_mapping = load_model_and_tokenizer()

if model and tokenizer and label_mapping:
    st.title("Sentiment Analysis App")

    user_input = st.text_area("Enter text for sentiment analysis:", "")

    if st.button("Analyze Sentiment"):
        if user_input:
            # Preprocess the input text (you might need to adapt this based on your model\'s preprocessing)
            # For an LSTM, typically tokenization and padding are needed
            # Ensure the tokenizer is fitted on the same vocabulary as during training
            # and the max_len matches your model\'s input shape.

            # Example preprocessing (adjust as per your model\'s requirements):
            # Assuming your tokenizer expects text and outputs sequences
            # And your model expects padded sequences
            sequence = tokenizer.texts_to_sequences([user_input])
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=model.input_shape[1])

            # Make prediction
            prediction = model.predict(padded_sequence)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Map prediction to sentiment label
            sentiment_labels = {v: k for k, v in label_mapping.items()}
            predicted_sentiment = sentiment_labels.get(predicted_class, "Unknown")

            st.write(f"Sentiment: **{predicted_sentiment}**")
        else:
            st.warning("Please enter some text to analyze.")
else:
    st.warning("Model could not be loaded. Please check the logs above for details.")


