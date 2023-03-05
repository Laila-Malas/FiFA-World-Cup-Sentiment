#import Necessary Libraries
from facebook_scraper import get_posts
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Start to collect Facebook posts data by facebook_scraper Library
from w3lib.html import remove_tags

listposts = []
for post in get_posts("fifaworldcup",pages=110,extra_info=True,options={ "comments" : True,
                "comments" : "generator",
                "allow_extra_requests": True,
                "comment_reactors": True,
                "reactions": True,},cookies="facebook.com_cookies.txt"):
    print(post['text'][:50])
    listposts.append(post)
#Check Data Frame information
df=pd.DataFrame(listposts)
print(df.info())
#Display Header of data frame
print(df.head())
#save Dataframe into CSV file
df.to_csv('blank.csv',index=False)

# Perform basic line plot to visualize post datetime Vs post
fig,ax=plt.subplots(figsize=(20,10))
ax.plot(df['time'],df['likes'],marker="o")
from matplotlib.dates import  DateFormatter
ax.xaxis.set_major_formatter(DateFormatter('%d-%m-%y %H:%M'))

# Perform basic line plot to visualize post datetime vs likes, shares and comments
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(df['time'], df['likes'], label = "Likes", marker="o")
ax.plot(df['time'], df['shares'], label = "Shares", marker="s")
ax.plot(df['time'], df['comments'], label = "Comments", marker="*")
plt.legend()
from matplotlib.dates import DateFormatter
ax.xaxis.set_major_formatter(DateFormatter('%d-%m-%y %H:%M'))

#Reload DataFrame  from CSV file
data=pd.read_csv("fifa.csv")
data['time'] = pd.to_datetime(data['time'])
print(data.head())
# Check dataframe information, 47 data columns in total, no change
print(data.info())
print(data.shape)
# Select and display post_id & reactions columns. In reactions column, there are various type of reactions
# in json/dictionary format.
print(data[['post_id', 'reactions']])
# Expand reactions columns into multiple columns
data['reactions'] = data['reactions'].apply(lambda x : dict(eval(x)) )
post_df_full_csv_reactions = data['reactions'].apply(pd.Series )
# Merge expanded columns into dataframe
post_df_full_csv_with_reactions = pd.concat([data, post_df_full_csv_reactions], axis=1).drop('reactions', axis=1)
# Display header of dataframe
post_df_full_csv_with_reactions.head()
# Check dataframe information, 53 data columns in total, 6 columns are being added
print(post_df_full_csv_with_reactions.info())
# Perform basic line plot to visualize post datetime vs Like, Love, Ha Ha, Wow, Sad, Angry,
# Care, shares and comments
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(post_df_full_csv_with_reactions['time'], post_df_full_csv_with_reactions['like'], label = "Like", marker="o")
ax.plot(post_df_full_csv_with_reactions['time'], post_df_full_csv_with_reactions['love'], label = "Love", marker="o")
ax.plot(post_df_full_csv_with_reactions['time'], post_df_full_csv_with_reactions['haha'], label = "Ha Ha", marker="o")
ax.plot(post_df_full_csv_with_reactions['time'], post_df_full_csv_with_reactions['wow'], label = "Wow", marker="o")
ax.plot(post_df_full_csv_with_reactions['time'], post_df_full_csv_with_reactions['sad'], label = "Sad", marker="o")
ax.plot(post_df_full_csv_with_reactions['time'], post_df_full_csv_with_reactions['angry'], label = "Angry", marker="o")
ax.plot(post_df_full_csv_with_reactions['time'], post_df_full_csv_with_reactions['care'], label = "Care", marker="o")
ax.plot(post_df_full_csv_with_reactions['time'], post_df_full_csv_with_reactions['shares'], label = "Shares", marker="s")
ax.plot(post_df_full_csv_with_reactions['time'], post_df_full_csv_with_reactions['comments'], label = "Comments", marker="*")
plt.legend()
from matplotlib.dates import DateFormatter
ax.xaxis.set_major_formatter(DateFormatter('%d-%m-%y %H:%M'))

# Select and display post_text & Like, Love, Ha Ha, Wow, Sad, Angry, Care, Shares and Comments columns.
print(post_df_full_csv_with_reactions[['post_text','like','love','haha','wow','sad','angry','care',
                                 'shares','comments']])
# Display Like, Love, Ha Ha, Wow, Sad, Angry, Care columns relationship with Shares
print(post_df_full_csv_with_reactions[['like','love','haha','wow','sad','angry','care']].corrwith(post_df_full_csv_with_reactions['shares']))
# Display Like, Love, Ha Ha, Wow, Sad, Angry, Care columns relationship with Comments
print(post_df_full_csv_with_reactions[['like','love','haha','wow','sad','angry','care']].corrwith(post_df_full_csv_with_reactions['comments']))

 # import necessary Library
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
#Define the Service Key and endpoint of Azure Text Analytics
key="1e7f07391222403f8f238a4153c89202"
endpoint="https://op890.cognitiveservices.azure.com/"
# Configure Azure Text Analytics client library
ta_credential = AzureKeyCredential(key)
text_analytics_client = TextAnalyticsClient(
        endpoint=endpoint,
        credential=ta_credential)
client = text_analytics_client

post_df_post_text_sentiment = []
key_phrase_extract = []

# Pass post text content to Azure Text Analytics and collect sentiment result
for index, headers in post_df_full_csv_with_reactions.iterrows():
    post_text_content = str(headers['post_text'])
    print("Post Text Content: {}".format(post_text_content))
    documents = [post_text_content]
    response = client.analyze_sentiment(documents=documents, language="en")[0]
    sentiment = response.sentiment
    print("Post Text Content Sentiment: {}".format(sentiment))
    like = str(headers['like'])
    print("Number of Like: {}".format(like))
    love = str(headers['love'])
    print("Number of Love: {}".format(love))
    haha = str(headers['haha'])
    print("Number of Ha Ha: {}".format(haha))
    wow = str(headers['wow'])
    print("Number of WoW: {}".format(wow))
    sad = str(headers['sad'])
    print("Number of Sad: {}".format(sad))
    angry = str(headers['angry'])
    print("Number of Angry: {}".format(angry))
    care = str(headers['care'])
    print("Number of Care: {}".format(care))
    shares = str(headers['shares'])
    print("Number of Shares: {}".format(shares))
    comments = str(headers['comments'])
    print("Number of Comments: {}".format(comments))
    response_ekp = client.extract_key_phrases(documents=documents)[0]
    for phrase in response_ekp.key_phrases:
        # print("Key phrase: {}".format(phrase))
        key_phrase_extract.append(phrase)
        print(key_phrase_extract)

    post_df_post_text_sentiment.append([post_text_content, sentiment, like, love, haha, wow,
                                        sad, angry, care, shares, comments, key_phrase_extract])

    key_phrase_extract = []

# Convert collected post text content with sentiment to Pandas dataframes.
post_df_post_text_sentiment = pd.DataFrame(post_df_post_text_sentiment, columns=['post_text', 'sentiment',
                                                                                 'like', 'love', 'haha',
                                                                                 'wow', 'sad', 'angry',
                                                                                 'care', 'shares', 'comments',
                                                                                 'key_phrase_extract'])

# Create new dataframe to perform factorization
post_df_post_text_sentiment_factorized = post_df_post_text_sentiment.copy()
# Perform factorization for sentiment column
post_df_post_text_sentiment_factorized.sentiment = pd.factorize(post_df_post_text_sentiment_factorized.sentiment)[0]
# Change columns back to correct datatype
post_df_post_text_sentiment_factorized['like'] = post_df_post_text_sentiment_factorized['like'].fillna(0).astype('float64')
post_df_post_text_sentiment_factorized['love'] = post_df_post_text_sentiment_factorized['love'].fillna(0).astype('float64')
post_df_post_text_sentiment_factorized['haha'] = post_df_post_text_sentiment_factorized['haha'].fillna(0).astype('float64')
post_df_post_text_sentiment_factorized['wow'] = post_df_post_text_sentiment_factorized['wow'].fillna(0).astype('float64')
post_df_post_text_sentiment_factorized['sad'] = post_df_post_text_sentiment_factorized['sad'].fillna(0).astype('float64')
post_df_post_text_sentiment_factorized['angry'] = post_df_post_text_sentiment_factorized['angry'].fillna(0).astype('float64')
post_df_post_text_sentiment_factorized['care'] = post_df_post_text_sentiment_factorized['care'].fillna(0).astype('float64')
post_df_post_text_sentiment_factorized['shares'] = post_df_post_text_sentiment_factorized['shares'].fillna(0).astype('float64')
post_df_post_text_sentiment_factorized['comments'] = post_df_post_text_sentiment_factorized['comments'].fillna(0).astype('float64')
# Display factorized sentiment dataframe, 0 = Negative, 1 = Neutral, 2 = Positive
print(post_df_post_text_sentiment_factorized)
post_df_post_text_sentiment_factorized.to_csv("Sentiment2.csv")
#plotting Sentiments and like columns relationship by seaborn
fig,ax=plt.subplots()
fig.set_size_inches(12,8)
plt.title('Statistical Analysis by Sentiment Score and Number of like',fontsize=10)
sns.regplot(x='sentiment',y='like',data=post_df_post_text_sentiment_factorized)
sns.despine()
#plotting Sentiment & shares columns relationship by seaborn
fig,ax=plt.subplots()
fig.set_size_inches(12,8)
plt.title('Statistical Analysis by Sentiment Score and Number of Shares', fontsize=10)
sns.regplot(x='sentiment', y= 'shares', data=post_df_post_text_sentiment_factorized)
sns.despine()
# Plotting sentiment & Comments columns relationship by Seaborn
fig, ax = plt.subplots()
fig.set_size_inches(12, 8)
plt.title('Statistical Analysis by Sentiment Score and Number of Comments', fontsize=10)
sns.regplot(x='sentiment', y= 'comments', data=post_df_post_text_sentiment_factorized)
sns.despine()
# import Worldcloud Library for worldcloud generations
from wordcloud import WordCloud
text=post_df_post_text_sentiment_factorized['key_phrase_extract'].values
text_a = str(text).replace('list', '')

wordcloud = WordCloud(width = 500, height = 500,
            background_color ='white',
            min_font_size = 10).generate(str(text_a))

plt.figure(figsize = (30, 10), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

#Build Multi Classification Neural Network
import numpy as np
import pandas as pd
data=pd.read_csv("Sentiment.csv")
y=data['sentiment']
x=data['post_text']
print(x.shape)
print(y.shape)

# work with labels
# encode class values as integers
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

from keras.utils import np_utils
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

print(encoded_Y)
print(dummy_y)
from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer()
x=count.fit_transform(x)
x=x.toarray()
print(x.shape)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,dummy_y,test_size=0.10)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
from keras.models import Sequential
from keras.layers import Dense,Dropout
model = Sequential()
model.add(Dense(units=3000,activation='relu',input_shape=(1310,5716)))
model.add(Dropout(rate=0.1))
model.add(Dense(units=4,activation='softmax'))
model.summary()

# compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy', # this is different instead of binary_crossentropy (for regular classification)
              metrics=['accuracy'])


model.fit(x_train,
                y_train,
                    epochs=50, # you can set this to a big number!
                    batch_size=10,
                    verbose=1)

y_pred=model.predict(x_test)
from sklearn import metrics

print(metrics.confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1)))
cm=metrics.confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1))
print('Accuracy: ' + str(np.round(100*float(cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3])/float((cm[0][0]+cm[0][1]+cm[0][2]+cm[0][3]+cm[1][0]+cm[1][1]+cm[1][2]+cm[1][3]+cm[2][0]+cm[2][1]+cm[2][2]+cm[2][3]+cm[3][0]+cm[3][1]+cm[3][2]+cm[3][3])),2))+'%')
print(metrics.accuracy_score(y_test.argmax(axis=1),y_pred.argmax(axis=1)))
acc=metrics.accuracy_score(y_test.argmax(axis=1),y_pred.argmax(axis=1))
# Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
cm_df = pd.DataFrame(cm,
                     index = ['positive','negative'],
                     columns = ['positive','negative'])

# Plotting the confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

from keras.wrappers.scikit_learn \
    import KerasClassifier
from sklearn.model_selection import GridSearchCV
#grid Search(Define search space as a grid of hyperparameter values and evaluate every position in the grid)
def build_cls(optimizer):
    model = Sequential()
    model.add(Dense(units=100, activation='relu', input_shape=(1296, 4943)))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=3298, activation='relu'))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=4, activation='softmax'))
    model.summary()

    # compile the model
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  # this is different instead of binary_crossentropy (for regular classification)
                  metrics=['accuracy'])
    return model

gs_clf=KerasClassifier(build_fn=build_cls)
params={'batch_size':[10,25,50],
        'epochs':[50,100,200],
        'optimizer':['adam','SGD']}
#Keras Classifier : Compatability between keras and sklearn
gs=GridSearchCV(estimator=gs_clf,param_grid=params,scoring='accuracy',cv=10)
gs=gs.fit(x_train,y_train)

print(gs.best_params_)
print(gs.best_score_)



# Build Prediction Deep Learning Model
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
from keras.layers import Flatten,GlobalAvgPool1D,Embedding,Conv1D,LSTM
from sklearn.model_selection import train_test_split
sentiment=pd.DataFrame(post_df_post_text_sentiment_factorized,columns=['post_text','sentiment'])
sentiment.to_csv("Sentiment2.csv")
print(sentiment.shape)
print(sentiment.isnull().values.any())
sns.countplot(x="sentiment",data=sentiment)
TAG_RE=re.compile(r'<[^>]+>')
def remove_tags(text):


    return TAG_RE.sub('',text)
import nltk
nltk.download('stopwords')

class CustomPreprocess():
    '''Cleans text data up, leaving only 2 or more char long non-stepwords composed of A-Z & a-z only
    in lowercase'''

    def __init__(self):
        pass

def preprocess_text(self, sen, stopwords_list=None):
    sen = sen.lower()

    # Remove html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ',
                      sentence)  # When we remove apostrophe from the word "Mark's", the apostrophe is replaced by an empty space. Hence, we are left with single character "s" that we are removing here.

    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ',
                      sentence)  # Next, we remove all the single characters and replace it by a space which creates multiple spaces in our text. Finally, we remove the multiple spaces from our text as well.

    # Remove Stopwords
    pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s*')
    sentence = pattern.sub('', sentence)

