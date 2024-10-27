import streamlit as st
from transformers import pipeline
# from transformers import *

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs from TensorFlow
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations to avoid specific informational message
# os.environ['TRANSFORMERS_CACHE'] = "C:/Users/Makled/.cache"

st.header("Emoji recognition")

emoji_mapping = {
    'joy': '😊',
    'sadness': '😢',
    'anger': '😠',
    'fear': '😨',
    'surprise': '😮',
    'disgust': '🤢'
}


classifier1 = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
classifier2 = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
# classifier3 = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion")
# classifier4 = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-emotion")
classifier5 = pipeline("text-classification", model="monologg/bert-base-cased-goemotions-original")
classifier6 = pipeline("text-classification", model="michellejieli/emotion_text_classifier")
# classifier7 = pipeline("text-classification", model="arpanghoshal/EmoBERTa")

sentence = st.text_input("Express what you feel")

result1 = classifier1(sentence)
result2 = classifier2(sentence)
# result3 = classifier3(sentence)
# result4 = classifier4(sentence)
result5 = classifier5(sentence)
result6 = classifier6(sentence)
# result7 = classifier7(sentence)

emotion1 = max(result1, key=lambda x: x['score'])
emotion2 = max(result2, key=lambda x: x['score'])
# emotion3 = max(result3, key=lambda x: x['score'])
# emotion4 = max(result4, key=lambda x: x['score'])
emotion5 = max(result5, key=lambda x: x['score'])
emotion6 = max(result6, key=lambda x: x['score'])
# emotion7 = max(result7, key=lambda x: x['score'])

emoji1 = emoji_mapping.get(emotion1['label'], '🤷‍♂️')
emoji2 = emoji_mapping.get(emotion2['label'], '🤷‍♂️')
# emoji3 = emoji_mapping.get(emotion3['label'], '🤷‍♂️')
# emoji4 = emoji_mapping.get(emotion4['label'], '🤷‍♂️')
emoji5 = emoji_mapping.get(emotion5['label'], '🤷‍♂️')
emoji6 = emoji_mapping.get(emotion6['label'], '🤷‍♂️')
# emoji7 = emoji_mapping.get(emotion7['label'], '🤷‍♂️')

st.markdown(f"<h3 style='text-align: center;'>{emoji1}{emoji2}{emoji5}{emoji6}</h3>", unsafe_allow_html=True)

# st.write("Emotion Scores:")

st.write(result1 + result2 + result5 + result6)

st.text("by Ahmed Makled  :)")
