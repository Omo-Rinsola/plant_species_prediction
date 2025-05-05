import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Load Model
model = tf.keras.models.load_model("best_model.h5")

# Function for image prediction:
def predict_image(img):
    img = img.resize((256, 256))  # Resize the image to match the model's input size
    img_array = np.array(img) / 255.0  # Scale the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return prediction

st.title('House Plant Species Classifier')
st.write("Upload a picture of a house plant, and we'll predict its species!")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Make prediction
    prediction = predict_image(img)
    
    # List of plant species
   
    class_names = [
    'African Violet (Saintpaulia ionantha)', 'Aloe Vera', 'Anthurium (Anthurium andraeanum)', 
    'Areca Palm (Dypsis lutescens)', 'Asparagus Fern (Asparagus setaceus)', 'Begonia (Begonia spp.)', 
    'Bird of Paradise (Strelitzia reginae)', 'Birds Nest Fern (Asplenium nidus)', 'Boston Fern (Nephrolepis exaltata)', 
    'Calathea', 'Cast Iron Plant (Aspidistra elatior)', 'Chinese Money Plant (Pilea peperomioides)', 
    'Chinese evergreen (Aglaonema)', 'Christmas Cactus (Schlumbergera bridgesii)', 'Chrysanthemum', 
    'Ctenanthe', 'Daffodils (Narcissus spp.)', 'Dracaena', 'Dumb Cane (Dieffenbachia spp.)', 
    'Elephant Ear (Alocasia spp.)', 'English Ivy (Hedera helix)', 'Hyacinth (Hyacinthus orientalis)', 
    'Iron Cross begonia (Begonia masoniana)', 'Jade plant (Crassula ovata)', 'Kalanchoe', 
    'Lilium (Hemerocallis)', 'Lily of the valley (Convallaria majalis)', 'Money Tree (Pachira aquatica)', 
    'Monstera Deliciosa (Monstera deliciosa)', 'Orchid', 'Parlor Palm (Chamaedorea elegans)', 'Peace lily', 
    'Poinsettia (Euphorbia pulcherrima)', 'Polka Dot Plant (Hypoestes phyllostachya)', 'Ponytail Palm (Beaucarnea recurvata)', 
    'Pothos (Ivy arum)', 'Prayer Plant (Maranta leuconeura)', 'Rattlesnake Plant (Calathea lancifolia)', 
    'Rubber Plant (Ficus elastica)', 'Sago Palm (Cycas revoluta)', 'Schefflera', 'Snake plant (Sanseviera)', 
    'Tradescantia', 'Tulip', 'Venus Flytrap', 'Yucca', 'ZZ Plant (Zamioculcas)'
]

    predicted_class = class_names[np.argmax(prediction)]
    
    st.write(f"Prediction: {predicted_class}")