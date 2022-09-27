import pandas as pd
from nltk.corpus import stopwords
import string


ds =pd.read_csv("spam.csv",encoding='latin1')
# print(ds.head())


# dropping NAN columns
ds.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# print(ds.head())


# Lower case letters
def lowercase_text(text):
    text=text.lower()
    return text
ds['v2']=ds['v2'].apply(lowercase_text)
# print(ds.head())


#Removing puctuation
PUNCT_TO_REMOVE=string.punctuation
def remove_punctuation(text):
    text=text.translate(str.maketrans('','',PUNCT_TO_REMOVE))
    return text
ds['v2']=ds['v2'].apply(remove_punctuation)
# print(ds.head())


# Removing Stopwords
STOPWORDS=set(stopwords.words('english'))
def remove_stopwords(text):
    text= " ".join([word for word in str(text).split() if word not in STOPWORDS])
    return text
ds['v2']=ds['v2'].apply(remove_stopwords)
# print(ds.head())

# unique words
tokens=ds['v2'].str.split()
def unique(sequence):
    seen=set()
    return [x for x in sequence if not (x in seen or seen.add(x))]
list=tokens.apply(unique)
# print(list.head())

spamlist=ds['v2'][ds['v1']=='spam']
hamlist=ds['v2'][ds['v1']=='ham']
# print(spamlist.head())
# print(hamlist.head())

spamlist.str.split().apply(unique)
spamlist.str.split(' ', expand=True).stack().value_counts()
hamlist.str.split().apply(unique)
hamlist.str.split(' ', expand=True).stack().value_counts()

# print(spamlist.str.split().apply(unique))
# print(spamlist.str.split(' ', expand=True).stack().value_counts())
# print(hamlist.str.split().apply(unique))
# print(hamlist.str.split(' ', expand=True).stack().value_counts())

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(ds.v2)
y = ds.v1

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
from sklearn.linear_model import LogisticRegression
regression = LogisticRegression()
regression.fit(x_train,y_train)
y_pred = regression.predict(x_test)
print(pd.crosstab(y_test, y_pred, rownames = ['Truth'], colnames =['Predicted'], margins = True))
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))











