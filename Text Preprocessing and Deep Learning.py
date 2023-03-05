#importing Essential Librarires
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from numpy import array
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation,Dropout,Dense
from keras.layers import Flatten, GlobalMaxPool1D, Embedding, Conv1D, LSTM, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
#Loading Dataset
facebook_data = pd.read_csv("Sentiment.csv")
print(facebook_data.shape)
print(facebook_data.head(5))
facebook_data['post_text']=facebook_data['post_text'].astype(str)
#Checking for Missing Values
facebook_data.isnull().values.any()

# Let's observe distribution of positive / negative /neutral and mixed sentiments in dataset
import seaborn as sns
sns.countplot(x="sentiment", data=facebook_data)
#Data Preprocessing
print(facebook_data["post_text"][2])
TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    '''Removes HTML tags: replaces anything between opening and closing <> with empty space'''

    return TAG_RE.sub('', text)
import nltk
nltk.download('stopwords')


def preprocess_text(sen):
    '''Cleans text data up, leaving only 2 or more char long non-stepwords composed of A-Z & a-z only
    in lowercase'''

    sentence = sen.lower()

    # Remove html tags
    sentence = remove_tags(sentence)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ',
                      sentence)  # When we remove apostrophe from the word "Mark's", the apostrophe is replaced by an empty space. Hence, we are left with single character "s" that we are removing here.

    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ',
                      sentence)  # Next, we remove all the single characters and replace it by a space which creates multiple spaces in our text. Finally, we remove the multiple spaces from our text as well.

    # Remove Stopwords
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    sentence = pattern.sub('', sentence)

    return sentence

# Calling preprocessing_text function on movie_reviews

X = []
sentences = list(facebook_data['post_text'])
for sen in sentences:
    X.append(preprocess_text(sen))
print(X[2])
# Converting sentiment labels to 0 & 1
X=np.array(X)
y = facebook_data['sentiment']

y = np.array(list(map(lambda x: 1 if x=="positive" or x=="neutral" else 0, y)))
import seaborn as sns
sns.countplot(x=y, data=facebook_data)

# Split Dataset to Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# The train set will be used to train our deep learning models
# while test set will be used to evaluate how well our model performs
#Preparing the Embedding Layer
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(X_train)

X_train = word_tokenizer.texts_to_sequences(X_train)
X_test = word_tokenizer.texts_to_sequences(X_test)
# Adding 1 to store dimensions for words for which no pretrained word embeddings exist

vocab_length = len(word_tokenizer.word_index) + 1

vocab_length

# Padding all reviews to fixed length 100

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
# Load GloVe word embeddings and create an Embeddings Dictionary

from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('a2_glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()
# Create Embedding Matrix having 100 columns
# Containing 100-dimensional GloVe word embeddings for all words in our corpus.

embedding_matrix = zeros((vocab_length, 100))
for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

print(embedding_matrix.shape)

#RNN with LSTM Neural Network Model
from keras.layers import LSTM
# Neural Network architecture

lstm_model = Sequential()
embedding_layer = Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)

lstm_model.add(embedding_layer)
lstm_model.add(LSTM(128))

lstm_model.add(Dense(1, activation='sigmoid'))
# Model compiling

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(lstm_model.summary())
lstm_model_history = lstm_model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
score = lstm_model.evaluate(X_test, y_test, verbose=1)
Y_pred=lstm_model.predict(X_test)
ycp=Y_pred>0.5

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,ycp)
print(cm)
cm_df = pd.DataFrame(cm,
                     index = ['positive','negative'],
                     columns = ['positive','negative'])
import matplotlib.pyplot as plt
# Plotting the confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()
# Model Performance
print("Test Score:", score[0])
print("Test Accuracy:", score[1])
import matplotlib.pyplot as plt

plt.plot(lstm_model_history['accuracy'])
plt.plot(lstm_model_history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(lstm_model_history['loss'])
plt.plot(lstm_model_history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

print('Accuracy: ' + str(np.round(100*float(cm[0][0]+cm[1][1])/float((cm[0][0]+cm[1][1] + cm[1][0] + cm[0][1])),2))+'%')
print('Precsion: ' + str(np.round(100*float((cm[1][1]))/float((cm[0][1]+cm[1][1])),2))+'%')
print('Recall: ' + str(np.round(100*float((cm[1][1]))/float((cm[1][0]+cm[1][1])),2))+'%')
p=(cm[1][1])/((cm[0][1]+cm[1][1]))
r=(cm[1][1])/((cm[1][0]+cm[1][1]))
f1=2*(p*r)/(p+r)
print("F1-score:",f1*100)
#Actual Vs Predicted table
error=pd.DataFrame(np.array(y_test).flatten(),columns=['actual'])
error['predicted']=np.array(Y_pred)
print(error)