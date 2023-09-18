import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image as im
import PIL 
#%matplotlib inline
num_classes = 10
input_shape = (28, 28, 1)

objt=tf.keras.datasets.mnist
(X_train, y_train), (X_test,y_test)=objt.load_data()


# X_train = X_train.astype("float32") / 255
# X_test = X_test.astype("float32") / 255
# # Make sure images have shape (28, 28, 1)
# X_train = np.expand_dims(X_train, -1)
# X_test = np.expand_dims(X_test, -1)
# print("x_train shape:", X_train.shape)
# print(X_train.shape[0], "train samples")
# print(X_test.shape[0], "test samples")


# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

# X_new=np.concatenate((X_train, X_test))
# y_new=np.concatenate((y_train, y_test))
# print(X_train.shape)
# print(X_new.shape)
# print(y_new.shape)

# print(y_train)

# model = keras.Sequential(
#     [
#         keras.Input(shape=input_shape),
#         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation="softmax"),
#     ]
# )

# model.summary()

# batch_size = 128
# epochs = 15

# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# model.fit(X_new, y_new, batch_size=batch_size, epochs=epochs, validation_split=0.1)
# model.save("keras_digit_test_include.h5")

# score = model.evaluate(X_test, y_test, verbose=0)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])

# loaded_model = keras.models.load_model('keras_digit_accurate.h5')
# score = loaded_model.evaluate(X_test, y_test, verbose=0)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])

#................................................................................................


# for i in range(9):
#     plt.subplot(330+1+i)
#     plt.imshow(X_train[i])
#     plt.show()

# X_train=X_train/255.0
# X_test=X_test/255.0

# model=tf.keras.models.Sequential([Flatten(input_shape=(28,28)),
                                        
#                                         Dense(650,activation='relu'),
                                        
#                                         Dense(450,activation='relu'),
                                        
#                                         Dense(250,activation='relu'),
                                        
#                                         Dense(150,activation='relu'),
                                        
#                                         Dense(10,activation=tf.nn.softmax)])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(X_train,y_train, epochs=10)
# model.save("keras_digit_temp.h5")
# test=X_test[0].reshape(-1,28,28)
# predicted=model.predict(test)
# print(predicted)

#count=0

def predict_digit(img):
    if img is not None:

        loaded_model = keras.models.load_model('keras_digit_test_include.h5')

        
        #img_data = im.fromarray(img)
        #img_data.save(f"image1.jpg")
        #count=count+1
        img_3d=img.reshape(-1,28,28)
        img_resized=img_3d/255.0
        pred_prob=loaded_model.predict(img_resized)

        pred_prob=pred_prob*100

        print((pred_prob))
        

        simple = pd.DataFrame(
        {
        "a": ["0", "1", "2", "3", "4", "5", "6", "7", "8","9"],
        "b": pred_prob[0], 
        }
        )

        predicted_val=np.argmax(pred_prob)
        return int(predicted_val), gr.BarPlot.update(
            simple,
            x="a",
            y="b",
            x_title="Digits",
            y_title="Identification Probabilities",
            title="Identification Probability",
            tooltip=["a", "b"],
            vertical=False,
            y_lim=[0, 100],
        )
        
    else:
        simple_empty = pd.DataFrame(
        {
        "a": ["0", "1", "2", "3", "4", "5", "6", "7", "8","9"],
        "b": [0,0,0,0,0,0,0,0,0,0],
        }
        )

        return " ", gr.BarPlot.update(
            simple_empty,
            x="a",
            y="b",
            x_title="Digits",
            y_title="Identification Probabilities",
            title="Identification Probability",
            tooltip=["a", "b"],
            vertical=False,
            y_lim=[0, 100],
            
        )
        

# iface=gr.Interface(prdict_digit, inputs='sketchpad', outputs=['label', gr.Slider(0,100, label='Probably 0'), gr.Slider(0,100, label='Probably 1')] ).launch()

# iface.launch(debug='true')

css='''
#title_head{
text-align: center;
text-weight: bold;
text-size:30px;
}
#name_head{
text-align: center;
}
'''

with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("<h1>Digit Identifier</h1>", elem_id='title_head')
            gr.Markdown("<h2>By Alok</h2>", elem_id="name_head")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                skch=gr.Sketchpad()
            with gr.Row():
                with gr.Column():
                    clear=gr.ClearButton(skch)
                with gr.Column():
                    btn=gr.Button("Identify")
                    
        with gr.Column():
            gr.Markdown("Identified digit")
            label=gr.Label("")
            gr.Markdown("Other possible values")
            bar = gr.BarPlot()
    btn.click(predict_digit,inputs=skch,outputs=[label,bar])

           
        
    
demo.launch()
    


