import os
import pandas as pd
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import streamlit as st

# Set TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' to suppress all messages

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Create VGG16 model for feature extraction
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Load styles DataFrame
styles_df = pd.read_csv('E:/College Material/Fall Minimester 3/BDA/archive/fashion-dataset/styles.csv')

# Function to recommend products
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    try:
        # Create 'uploads' directory if it doesn't exist
        os.makedirs('uploads', exist_ok=True)
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return 0

# Function for feature extraction
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function to display image details
def display_image_details(file_path, column):
    image_id = os.path.basename(file_path).split('.')[0]

    # Check if image_id exists in styles_df
    if int(image_id) in styles_df['id'].values:
        image_details = styles_df[styles_df['id'] == int(image_id)].iloc[0]

        with column:
            st.image(file_path)
            st.caption(f"Product ID: {image_id}")
            #st.write(f"**Name:** {image_details['productDisplayName']}")
            #st.write(f"\n\n**Category:** {image_details['subCategory']}")
            
        return {
            'Product ID': image_id,
            'Name': image_details['productDisplayName'],
            'Category': image_details['subCategory']
        }

    else:
        with column:
            st.image(file_path)
            st.caption(f"Product ID: {image_id}")
            #st.warning("No information found for this product.")
        return None

# Streamlit app title
st.title('Product Recommendation System')

# File upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Recommendation
        indices = recommend(features, feature_list)

        # Show image details
        product_info_list = []
        cols = st.columns(5)
        for i, col in enumerate(cols):
            file_path = filenames[indices[0][i]].replace("\\", "/")  # Replace backslashes with forward slashes
            info = display_image_details(file_path, col)
            if info:
                product_info_list.append(info)

        # Display a table with all product information
        if product_info_list:
            st.write("## All Products Information")
            all_products_table = pd.DataFrame(product_info_list)
            st.table(all_products_table)
        else:
            st.warning("No products details found.")
