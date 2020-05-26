#import libraries and packages
import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string

#create instances for stopwords and stemmer (portStemmer)
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

#Call the dataset
data = pd.read_csv("C:/Users/nisht/Anaconda3_n/envs/LL/NLP/Ch01/01_10/End/SMSSpamCollection.tsv", sep='\t')
data.columns = ['label', 'body_text']

#define a feature called punct%
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100
    
#create two features: body lengtha and number of punctuations in each text body
data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))

#pre-processing/ cleaning of data: tokenization, removing punctuations, converting text into lower case, stemming, removing stopwords
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text
    
#TF-IDF Vectorization and create a feature dataframe by concatinating all features
tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(data['body_text'])
X_tfidf_feat= pd.concat([data['body_len'], data['punct%'], pd.DataFrame(X_tfidf.toarray())], axis=1)
#Display:
X_tfidf_feat.head()

#Count Vectorization and create a feature dataframe by concatinating all features
count_vect = TfidfVectorizer(analyzer=clean_text)
X_count = tfidf_vect.fit_transform(data['body_text'])
X_count_feat= pd.concat([data['body_len'], data['punct%'], pd.DataFrame(X_tfidf.toarray())], axis=1)
#Display:
X_count_feat.head()

# Explore RandomForestClassifier Attributes & Hyperparameters
from sklearn.ensemble import RandomForestClassifier

# Explore RandomForestClassifier through Holdout Set
rf = RandomForestClassifier(n_estimators=50, max_depth=50, n_jobs=-1)
rf_model=rf.fit(X_train, y_train)
sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True)[0:10]
y_pred=rf_model.predict(X_test)
precision, recall, fscore, support=score(y_test, y_pred, pos_label='spam', average='binary')
print('Precision: {} / Recall: {} / Accuracy: {}'.format(round(precision,3),
     round(recall,3),
     (y_pred==y_test).sum()/len(y_pred)))
     
# Explore Random Forest model with grid search
def train_RF(n_est, depth):
    rf_gs= RandomForestClassifier(n_estimators=n_est, max_depth=depth, n_jobs=-1)
    rf_gs_model= rf_gs.fit(X_train, y_train)
    y_pred= rf_gs_model.predict(X_test)
    precision, recall, fscore, support=score(y_test, y_pred, pos_label='spam', average='binary')
    print('Est: {} / Depth: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(n_est, depth, round(precision,3), 
                                                                round(recall, 3), round((y_pred==y_test).sum() / len(y_pred),3)))
    
for n_est in [10, 50, 100]:
    for depth in [10, 20, 30, None]:
        train_RF(n_est, depth)
        
# Evaluate the Model with GridSearchCV for CountVector Data
from sklearn.model_selection import GridSearchCV
rf_gscv=RandomForestClassifier()
param= {'n_estimators': [10, 150, 300] ,
       'max_depth': [30, 60, 90, None] }

gs= GridSearchCV(rf_gscv, param, cv=5, n_jobs=-1)
gs_fit= gs.fit(X_tfidf_feat, data['label'])
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]

# Evaluate the Model with GridSearchCV for CountVector Data
rf_gscv=RandomForestClassifier()
param= {'n_estimators': [10,150,300] ,
       'max_depth': [30,60,90,None] }
       
gs= GridSearchCV(rf_gscv, param, cv=5, n_jobs=-1)
gs_fit= gs.fit(X_count_feat, data['label'])
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]
 
