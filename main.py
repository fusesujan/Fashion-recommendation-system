from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import joblib
import pickle
from PIL import Image
import os
import streamlit as st
import numpy as np
np.object = object
np.bool = bool
np.int = int  # Add these lines to resolve the deprecation warnings


feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = joblib.load('resnet50_for_feature_ext.pkl')

st.title('Fashion Recommender System')


def save_uploaded_file(uploaded_file):
    '''
    this function takes the file and save in local folder
    '''
    # Save the uploaded file in the current directory
    with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return 1


def feature_extraction(img_path, model):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array = np.array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(expanded_img_array).flatten()
    normalized_result = result / norm(result)
    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(
        n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


uploaded_file = st.file_uploader("Choose an image")
if uploaded_file:
    if save_uploaded_file(uploaded_file):
        # display the file
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=False)
        # # feature extract
        features = feature_extraction(os.path.join(
            "uploads", uploaded_file.name), model)
        # recommendention
        indices = recommend(features, feature_list)
        st.markdown("Here are some recommended Fashion Product")
        # show
        col1, col2, col3, col4, col5 = st.columns(5)
        RELATIVE_PATH = "Very-small-Dataset/"
        with col1:
            image = Image.open(RELATIVE_PATH+filenames[indices[0][0]])
            st.image(image)
        with col2:
            image = Image.open(RELATIVE_PATH+filenames[indices[0][1]])
            st.image(image)
        with col3:
            image = Image.open(RELATIVE_PATH+filenames[indices[0][2]])
            st.image(image)
        with col4:
            image = Image.open(RELATIVE_PATH+filenames[indices[0][3]])
            st.image(image)
        with col5:
            image = Image.open(RELATIVE_PATH+filenames[indices[0][4]])
            st.image(image)
    else:
        st.markdown("Some error occured in file upload")
