import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from dotenv import load_dotenv
load_dotenv()
import textwrap
import google.generativeai as genai
from IPython.display import Markdown


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def call_api(prompt,images_uploaded):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-pro-vision')
    i=[Image.open(i) for i in images_uploaded]
    responses = model.generate_content([prompt]+i)
    return  responses.text

        
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


st.title('Style Sensei System')
st.image("https://www.omnisend.com/blog/wp-content/uploads/2021/03/21-03-19-Fashion-ecommerce.jpg")

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('mmk/uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    
    except Exception as e:
        print("Error saving file:", e)
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# steps
# file upload -> save



#1st task
def similar():
    uploaded_file1 = st.file_uploader('option1',type=['png','jpeg','jpg'])
    
    if uploaded_file1 is not None:
            if save_uploaded_file(uploaded_file1):
            # display the file
                display_image = Image.open(uploaded_file1)
                st.image(display_image)
                # feature extract
                features = feature_extraction(os.path.join("mmk/uploads", uploaded_file1.name), model)

                # recommendention
                indices = recommend(features, feature_list)
                # show
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.image(filenames[indices[0][0]])
                with col2:
                    st.image(filenames[indices[0][1]])
                with col3:
                    st.image(filenames[indices[0][2]])
                with col4:
                    st.image(filenames[indices[0][3]])
                with col5:
                    st.image(filenames[indices[0][4]])
            else:
                st.header("Some error occured in file upload")
            
# 2nd task
def fashion():
    uploaded_file_multi=(st.file_uploader('option2',type=['png','jpeg','jpg'],accept_multiple_files=True))
    if uploaded_file_multi is not None:  
        for i in uploaded_file_multi:
            display_image = Image.open(i)
            st.image(display_image)
                          
        if len(uploaded_file_multi)>1:
                submit=st.button("GENERATE IF ALL IMAGES ARE UPLOADED")
                if submit:
                    response_submit=call_api("Based on the provided images of clothes,accessories and my face, can you help me determine if they complement each other well in terms of fashion?",uploaded_file_multi)
                    st.header("WHAT WE THINK ABOUT THIS ")
                    st.write(response_submit)
    
    
op=st.selectbox(("Kindly specify your choice among the options provided"),['Generate product images with similar characteristics','Ensure if the product aligns with your personal fashion flair','select'],index=2)
if op:
    if op=='Generate product images with similar characteristics':
        similar()
    elif op=='Ensure if the product aligns with your personal fashion flair':
        fashion()





st.sidebar.header("Shop By Department")


def my_widget():
    st.subheader('Hello there!')

# AND in st.sidebar!
with st.sidebar:
    my_expander = st.expander("Men's Fashion", expanded=True)
    my_expander.text('T-shirts')
    my_expander.text('shirts')
    my_expander.text('Jeans')
with st.sidebar:
    my_expander = st.expander("Accessiories", expanded=True)
    my_expander.text('Watches')
    my_expander.text('Bags and Luggages')
    my_expander.text('sunglasses')
with st.sidebar:
    my_expander = st.expander("Stores", expanded=True)
    my_expander.text('Sportswear')
    my_expander.text('The Designer Boutique')
    my_expander.text('Fashion sales and deals')

