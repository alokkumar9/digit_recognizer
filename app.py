import gradio as gr 
#import streamlit as st
from sklearn.neural_network import MLPClassifier 
import torchvision.datasets as datasets 
import seaborn as sns 
import pickle

#dark mode seaborn 
sns.set_style("darkgrid")

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

print(mnist_trainset.data.shape)
print(mnist_testset.data.shape)
print(mnist_trainset.targets.shape)
print(mnist_testset.targets.shape)
X_train = mnist_trainset.data
y_train = mnist_trainset.targets
X_test = mnist_testset.data
y_test = mnist_testset.targets

X_train = X_train.numpy() 
X_test = X_test.numpy()
y_train = y_train.numpy()
y_test = y_test.numpy()

X_train = X_train.reshape(60000, 784)/255.0
X_test = X_test.reshape(10000, 784)/255.0

#train the model 
mlp = MLPClassifier(hidden_layer_sizes=(50,50))
mlp.fit(X_train, y_train)

#print the accuracies 
print("Training Accuracy: ", mlp.score(X_train, y_train))
print("Testing Accuracy: ", mlp.score(X_test, y_test))

pickle.dump(mlp, open("digitmodel.sav", 'wb'))


loaded_model = pickle.load(open("digitmodel.sav", 'rb'))  #loding saved model

def predict(img):
    img = img.reshape(1,784)/255.0
    prediction = loaded_model.predict(img)[0]
    return int(prediction)

gr.Interface(fn= predict, inputs = "sketchpad", outputs ="label").launch()
#working on hface
