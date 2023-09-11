import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import pandas as pd
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

def predict_digit(img):
    if img is not None:

        loaded_model = keras.models.load_model('keras_digit_temp.h5')
    
        simple = pd.DataFrame(
        {
        "a": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        "b": [28, 55, 43, 91, 81, 53, 19, 87, 52],
        }
        )
        img_3d=img.reshape(-1,28,28)
        img_resized=img_3d/255.0
        pred_prob=loaded_model.predict(img_resized)
    
        pred_prob=pred_prob*100

        print((pred_prob))
        # prob0= 100*pred_prob[0]
        # prob1= 100*pred_prob[1]
        # prob2= 100*pred_prob[2]
        # prob3= 100*pred_prob[3]
        # prob4= 100*pred_prob[4]
        # prob5= 100*pred_prob[5]
        # prob6= 100*pred_prob[6]
        # prob7= 100*pred_prob[7]
        # prob8= 100*pred_prob[8]
        # prob9= 100*pred_prob[9]

        # print(prob2) 

    
        predicted_val=np.argmax(pred_prob)
        return int(predicted_val)
        
    else:
        return " "
        

# iface=gr.Interface(prdict_digit, inputs='sketchpad', outputs=['label', gr.Slider(0,100, label='Probably 0'), gr.Slider(0,100, label='Probably 1')] ).launch()

# iface.launch(debug='true')

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("Digit Identify", elem_id='title_head')
            gr.Markdown("By Alok")
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
            #bar=gr.BarPlot("Probability")
    btn.click(predict_digit,inputs=skch,outputs=label)

           
        
    
demo.launch()
    


