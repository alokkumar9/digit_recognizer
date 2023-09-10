import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
#%matplotlib inline


objt=tf.keras.datasets.mnist
(X_train, y_train), (X_test,y_test)=objt.load_data()

print(X_train.shape)

print(y_train)

# for i in range(9):
#     plt.subplot(330+1+i)
#     plt.imshow(X_train[i])
#     plt.show()

X_train=X_train/255.0
X_test=X_test/255.0

# model=tf.keras.models.Sequential([Flatten(input_shape=(28,28)),
#                                         Dense(784,activation='relu'),
#                                         Dense(650,activation='relu'),
#                                         Dense(550,activation='relu'),
#                                         Dense(450,activation='relu'),
#                                         Dense(350,activation='relu'),
#                                         Dense(256,activation='relu'),
#                                         Dense(128,activation='relu'),
#                                         Dense(80,activation='relu'),
#                                         Dense(40,activation='relu'),
                                        
#                                         Dense(10,activation=tf.nn.softmax)])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(X_train,y_train, epochs=10)
# model.save("keras_digit_temp.h5")
# test=X_test[0].reshape(-1,28,28)
# predicted=model.predict(test)
# print(predicted)

def prdict_digit(img):
    loaded_model = keras.models.load_model('keras_digit_temp.h5')
    img_3d=img.reshape(-1,28,28)
    img_resized=img_3d/255.0
    pred_prob=loaded_model.predict(img_resized)
    predicted_val=np.argmax(pred_prob)
    return int(predicted_val)

iface=gr.Interface(prdict_digit, inputs='sketchpad', outputs='label').launch()

iface.launch(debug='true')
    


