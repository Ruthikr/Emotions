import streamlit as st
import pickle
import numpy as np
import tensorflow
import keras
import math
from keras.utils import pad_sequences

model=pickle.load(open("emotions.pkl","rb"))

tokenizer=pickle.load(open("tokenizer.pkl","rb"))


st.title("Emotion detector")

text=st.text_area("enter your text")

if st.button("detect emotion"):
    text=tokenizer.texts_to_sequences([text])[0]
    text=np.array(text).reshape(1,-1)
    text=pad_sequences(text,maxlen=97,padding="post")
    result=model.predict(text)
    e=np.argmax(result)
    classes=["sadness","joy","love","anger","fear","surprise"]

    
    st.title("overall emotion")
   
    result=result.reshape(6)
    st.markdown("# {} - {}".format(classes[e],(str(math.floor(result[e]*100))+"%")))
    dictionary=dict()
    for i in range(len(classes)):
        dictionary[classes[i]]=(str(math.floor(result[i]*100))+"%")
        #st.write(classes[i],result[i])
    
    st.write(dictionary)
