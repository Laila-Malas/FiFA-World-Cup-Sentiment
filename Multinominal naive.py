import pandas as pd
import re
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import random
import sklearn.model_selection as model_selection
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from Test import word_tokenizer

df=pd.read_csv("Sentiment2.csv")
print(df.info())
print(df.drop_duplicates())
print(df.isnull().sum())
df = df[df.sentiment!=3]
df=df.dropna(axis=0)
# some basic pre-processing
# remove links
df['post_text'] = df['post_text'].apply(lambda x: re.sub(r"(www|http|https|pic)([a-zA-Z\.0-9:=\\~#/_\&%\?\-])*", ' ', x))
# remove mention symbol
df['post_text'] = df['post_text'].apply(lambda x: x.replace('@', ''))
# remove hashtag symbol
df['post_text'] = df['post_text'].apply(lambda x: x.replace('#', ''))
# convert all text to lower case (this helps in vectorization and training)
df['post_text'] = df['post_text'].apply(lambda x: x.lower())
X=df['post_text']
Y = (df["sentiment"])
import  numpy as np
Y=np.array(Y)
Y=Y.reshape(-1,1)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                    Y,
                                                                    train_size=0.90, test_size=0.10,random_state=101)
pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=0.0001, max_df=0.95, analyzer='word', ngram_range=(1, 3))),
        ('clf', MultinomialNB()),
    ])
# train the model
pipeline.fit(X_train, y_train)
feature_names = pipeline.named_steps['vect'].get_feature_names()
# test the model
y_predicted = pipeline.predict(X_test)
y_predicted=pd.get_dummies(y_predicted)
from sklearn import metrics
# print the classification report
print(metrics.classification_report(y_test, y_predicted))
print('# of features:', len(feature_names))
print('sample of features:', random.sample(feature_names, 40))
# calculate and print the model testing metrics
accuracy = accuracy_score(y_test, y_predicted)
precision = precision_score(y_test, y_predicted, average='weighted')
recall = recall_score(y_test, y_predicted, average='weighted')
f1 = f1_score(y_test, y_predicted, average='weighted')
print('Accuracy: ', "%.2f" % (accuracy*100))
print('Precision: ', "%.2f" % (precision*100))
print('Recall: ', "%.2f" % (recall*100))
print('F1: ', "%.2f" % (f1*100))
import numpy as np
error=pd.DataFrame(np.array(y_test).flatten(),columns=['actual'])
error['predicted']=np.array(y_predicted)
print(error)
from sklearn.linear_model import LogisticRegression
# using Label Powerset
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())
# train
classifier.fit(X_train, y_train)
# predict
predictions = classifier.predict(X_test,y_test
)
# accuracy
from sklearn.metrics import confusion_matrix
print("Accuracy = ",accuracy_score(y_test,y_predicted))
cm = confusion_matrix(y_test,y_predicted)
cm_df = pd.DataFrame(cm,
                     index = ['neutral','negative','positive'],
                     columns = ['neutral','negative','positive'])
import seaborn as sns
import matplotlib.pyplot as plt
#Plotting the confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()