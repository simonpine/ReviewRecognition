import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
nltk.download('stopwords')


tf_transformer = pickle.load(open('./transformModel_tf_transformer.pkl', 'rb'))
vcModel = pickle.load(open('./transformModel_vcModel.pkl', 'rb'))

stop_words = set(stopwords.words('english')) 
def removeStopwordsAndLower(text):
    
    words = text.lower().split() 
    filtered_words = [word for word in words if word not in stop_words] 
    return ' '.join(filtered_words)

def lemaAndStem(text):
    stemmer = SnowballStemmer("english")
    normalized_text = []
    for word in text.split():
        stemmed_word = stemmer.stem(word)
        normalized_text.append(stemmed_word)
    return ' '.join(normalized_text).replace(',', '')


def transformToPredict(textPredict):
    textPredict = removeStopwordsAndLower(textPredict)
    textPredict = lemaAndStem(textPredict)
    textPredict = vcModel.transform([textPredict])
    textPredict = tf_transformer.transform(textPredict)
    return textPredict

st.image('blog-hero_facebook-reviews-banner-group.jpg', caption='Banner')

st.write('''
# Review Recognition
         
This app recognizes if a review of a product is positive or negative, by using some NPL techniques.
         
''')

# st.write('Copy the review to know if it is good:')

text = st.text_area('Copy the review to know if it is good:') 

model = pickle.load(open('./trained_model_XGBoost.pkl', 'rb'))

st.button('Predict')

if(text == ''):
    st.write('waiting...')
elif(model.predict((transformToPredict(text)))[0] == 0):
    st.write('''##### ★☆☆☆☆ Negative 🤬 ''')
else:
    st.write('''##### ★★★★★ Positive 😀''')