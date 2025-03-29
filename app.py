from flask import Flask,redirect,url_for,render_template,request,jsonify
import math
import cv2
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import pickle
import io
from time import time
import random
import json
import tensorflow as tf
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling2D


app=Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ANNOTATION_FOLDER = "static/annotations"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ANNOTATION_FOLDER"] = ANNOTATION_FOLDER

import os
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret_key")

deepfake_model = load_model("ResNet50_Deepfake_Detector.h5")

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
gap_layer = GlobalAveragePooling2D()(base_model.output) 
feature_extractor = Model(inputs=base_model.input, outputs=gap_layer)

print(f"[INFO] Modified ResNet50 Output Shape: {feature_extractor.output_shape}")  

def preprocess_image(image):
    img = image.resize((224, 224))  
    img = np.array(img) / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img

def extract_features(img):
    img = preprocess_image(img)  
    features = feature_extractor.predict(img)  
    print(f"[DEBUG] Raw Extracted Feature Shape: {features.shape}")  
    features = features.reshape(1, -1) 
    print(f"[DEBUG] Flattened Feature Shape: {features.shape}")  
    return features 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index.html')
def index_return():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/deepfake.html')
def deepfake():
    return render_template('deepfake.html')

@app.route('/blogs.html')
def blogs():
    return render_template('blogs.html')

@app.route('/blog-post-1.html')
def blog1():
    return render_template('blog-post-1.html')

@app.route('/blog-post-2.html')
def blog2():
    return render_template('blog-post-2.html')

@app.route('/blog-post-3.html')
def blog3():
    return render_template('blog-post-3.html')

@app.route('/blog-post-4.html')
def blog4():
    return render_template('blog-post-4.html')

@app.route('/blog-post-5.html')
def blog5():
    return render_template('blog-post-5.html')

@app.route('/blog-post-6.html')
def blog6():
    return render_template('blog-post-6.html')


# Chatbot
nltk.download('popular')
lemmatizer = WordNetLemmatizer()
model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

app.static_folder = 'static'

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

# Deepfake detection
@app.route('/result', methods=['POST'])
def result():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    try:
        # Open and preprocess image
        image = Image.open(file)
        image = image.convert("RGB")

        # Save original uploaded image
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        image.save(file_path)

        # Preprocess & predict
        processed_image = preprocess_image(image)
        prediction = deepfake_model.predict(processed_image)
        pred_value = float(prediction)

        if pred_value > 0.5:
            result_text = "Real"
            confidence = pred_value
            annotated_filename = None  # No annotated image for real images
        else:
            result_text = "Fake"
            confidence = (1 - pred_value)

            # Define the expected annotated image path
            annotated_filename = "annotated_" + file.filename
            annotated_path = os.path.join(app.config["ANNOTATION_FOLDER"], annotated_filename)

            # Check if the manually saved annotated image exists
            if not os.path.exists(annotated_path):
                print(f"[WARNING] Annotated image not found: {annotated_path}")
                annotated_filename = None  # Do not pass an image if it doesn't exist

        return render_template(
            "result.html",
            data=result_text,
            confidence=confidence,
            uploaded_filename=file.filename,
            annotated_filename=annotated_filename
        )

    except Exception as e:
        print(f"[ERROR] Error processing image: {e}")
        return f"[ERROR] {str(e)}"

if __name__=='__main__':
    app.run(debug=True)