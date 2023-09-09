import gradio as gr 
#import streamlit as st
from sklearn.neural_network import MLPClassifier 
import torchvision.datasets as datasets 
import seaborn as sns 
import pickle

#dark mode seaborn 
sns.set_style("darkgrid")

loaded_model = pickle.load(open("digitmodel.sav", 'rb'))  #loding saved model

def predict(img):
    img = img.reshape(1,784)/255.0
    prediction = loaded_model.predict(img)[0]
    return int(prediction)

gr.Interface(fn= predict, inputs = "sketchpad", outputs ="label").launch()

