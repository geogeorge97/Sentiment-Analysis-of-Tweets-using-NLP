# %%
# Import necessary packages
import re
from os.path import join
import numpy as np
import pandas as pd
import nltk

# %%
# Define test sets
testsets = ['twitter-test1.txt', 'twitter-test2.txt', 'twitter-test3.txt']

# %%
# Skeleton: Evaluation code for the test sets
def read_test(testset):
    '''
    readin the testset and return a dictionary
    :param testset: str, the file name of the testset to compare
    '''
    id_gts = {}
    with open(testset, 'r', encoding='utf8') as fh:
        for line in fh:
            fields = line.split('\t')
            tweetid = fields[0]
            gt = fields[1]

            id_gts[tweetid] = gt

    return id_gts


def confusion(id_preds, testset, classifier):
    '''
    print the confusion matrix of {'positive', 'negative','neutral'} between preds and testset
    :param id_preds: a dictionary of predictions formated as {<tweetid>:<sentiment>, ... }
    :param testset: str, the file name of the testset to compare
    :classifier: str, the name of the classifier
    '''
    id_gts = read_test(testset)

    gts = []
    for m, c1 in id_gts.items():
        if c1 not in gts:
            gts.append(c1)

    gts = ['positive', 'negative', 'neutral']

    conf = {}
    for c1 in gts:
        conf[c1] = {}
        for c2 in gts:
            conf[c1][c2] = 0

    for tweetid, gt in id_gts.items():
        if tweetid in id_preds:
            pred = id_preds[tweetid]
        else:
            pred = 'neutral'
        conf[pred][gt] += 1

    print(''.ljust(12) + '  '.join(gts))

    for c1 in gts:
        print(c1.ljust(12), end='')
        for c2 in gts:
            if sum(conf[c1].values()) > 0:
                print('%.3f     ' % (conf[c1][c2] / float(sum(conf[c1].values()))), end='')
            else:
                print('0.000     ', end='')
        print('')

    print('')


def evaluate(id_preds, testset, classifier):
    '''
    print the macro-F1 score of {'positive', 'netative'} between preds and testset
    :param id_preds: a dictionary of predictions formated as {<tweetid>:<sentiment>, ... }
    :param testset: str, the file name of the testset to compare
    :classifier: str, the name of the classifier
    '''
    id_gts = read_test(testset)

    acc_by_class = {}
    for gt in ['positive', 'negative', 'neutral']:
        acc_by_class[gt] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

    catf1s = {}

    ok = 0
    for tweetid, gt in id_gts.items():
        if tweetid in id_preds:
            pred = id_preds[tweetid]
        else:
            pred = 'neutral'

        if gt == pred:
            ok += 1
            acc_by_class[gt]['tp'] += 1
        else:
            acc_by_class[gt]['fn'] += 1
            acc_by_class[pred]['fp'] += 1

    catcount = 0
    itemcount = 0
    macro = {'p': 0, 'r': 0, 'f1': 0}
    micro = {'p': 0, 'r': 0, 'f1': 0}
    semevalmacro = {'p': 0, 'r': 0, 'f1': 0}

    microtp = 0
    microfp = 0
    microtn = 0
    microfn = 0
    for cat, acc in acc_by_class.items():
        catcount += 1

        microtp += acc['tp']
        microfp += acc['fp']
        microtn += acc['tn']
        microfn += acc['fn']

        p = 0
        if (acc['tp'] + acc['fp']) > 0:
            p = float(acc['tp']) / (acc['tp'] + acc['fp'])

        r = 0
        if (acc['tp'] + acc['fn']) > 0:
            r = float(acc['tp']) / (acc['tp'] + acc['fn'])

        f1 = 0
        if (p + r) > 0:
            f1 = 2 * p * r / (p + r)

        catf1s[cat] = f1

        n = acc['tp'] + acc['fn']

        macro['p'] += p
        macro['r'] += r
        macro['f1'] += f1

        if cat in ['positive', 'negative']:
            semevalmacro['p'] += p
            semevalmacro['r'] += r
            semevalmacro['f1'] += f1

        itemcount += n

    micro['p'] = float(microtp) / float(microtp + microfp)
    micro['r'] = float(microtp) / float(microtp + microfn)
    micro['f1'] = 2 * float(micro['p']) * micro['r'] / float(micro['p'] + micro['r'])

    semevalmacrof1 = semevalmacro['f1'] / 2

    print(testset + ' (' + classifier + '): %.3f' % semevalmacrof1)

# %% [markdown]
# #### Load training set, dev set and testing set
# Here, you need to load the training set, the development set and the test set. For better classification results, you may need to preprocess tweets before sending them to the classifiers.

# %%
# Load training set, dev set and testing set
data = {}
tweetids = {}
tweetgts = {}
tweets = {}

for dataset in ['twitter-training-data.txt'] + testsets:
    data[dataset] = []
    tweets[dataset] = []
    tweetids[dataset] = []
    tweetgts[dataset] = []

# Reading in training dataset into df_tweets dataframe
df_tweets = pd.read_csv('twitter-training-data.txt',sep='\t',header=None)

# %%
#Rename columns
df_tweets.columns = ['id','sentiment','text']

# %% [markdown]
# ### Exploratory Data Analysis (EDA)

# %% [markdown]
# ## Text preprocessing

# %%
#Initialiazing TweetTokenizer to tokenize the text
from nltk.tokenize import TweetTokenizer

tweet_tokenizer = TweetTokenizer()

# %%
#Making a list of punctuation
import string
punc_list = list(string.punctuation)

# %%
#Making stopword list
from nltk.corpus import stopwords
stopwords_list = stopwords.words('english')

# %%
#Snowball stemmer
from nltk.stem.snowball import SnowballStemmer
snow_stemmer = SnowballStemmer(language='english')

# %%
def preprocess_text(text):
    
       
    #Lowercase the tweet
    text = text.lower()
    
    #Removing urls starting with http or https or www
    text = re.sub(r'https?:\/\/\S+','',text)
    text = re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)",'',text)
    
    #Removing html reference characters in the text like &amp;
    text = re.sub(r'&[a-z]+;','',text)

    #Remove @mentions and the text accompanying it
    text = re.sub(r"@\S*",'',text)

    #Remove non-alphanumeric as well as punctuation not used in emojis
    text = re.sub(r'[^A-Za-z0-9\(\):\-\s]',"",text)
    
    #Applying TweetTokenizer which handles emojis and hashtags
    text = tweet_tokenizer.tokenize(text)
    
    #Removing punctuation and stopwords, stemming
    punc_plus_stop_list = punc_list + stopwords_list
    text = [snow_stemmer.stem(token) for token in text if token not in punc_plus_stop_list]
    
    #Joining the tokens together to form readable text again
    text = ' '.join(text)   
    
    return text
    
    

# %%
#Preprocessed tweets stored in new column
df_tweets['clean_text'] = df_tweets['text'].apply(preprocess_text)

# %%
#Checking random example tweet
df_tweets['text'][41]

# %%
df_tweets['clean_text'][41]

# %% [markdown]
# * We can see that the hastags have been preserved( with spacing in between) and stemming has been done on the words. Irrelevant punctuation and stopwords have been removed, except for the punctuation symbols corresponding to emojis as they could potentially add value to our model.

# %%
df_tweets.head()

# %% [markdown]
# BOW for Naive Bayes

# %%
#Bag of words, set binary=True since we are using it for NaiveBayes which is a probabilistic model
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(binary=True,max_features=20000,ngram_range=(1,3))

# %%
X = bow.fit_transform(df_tweets['clean_text']).toarray()

# %%
y = df_tweets['sentiment']

# %% [markdown]
# Multinomial Naive Bayes

# %%
from sklearn.naive_bayes import MultinomialNB
clf_nb = MultinomialNB()

# %%
clf_nb.fit(X,y)

# %%
#Reading development data
df_dev = pd.read_csv('twitter-dev-data.txt',sep='\t',header=None)

# %%
df_dev.columns = ['id','sentiment','text']
df_dev['sentiment'] = df_dev['sentiment'].map(sent_category_dict)


# %%
#Cleaning dev data
df_dev['clean_text'] = df_dev['text'].apply(preprocess_text)

# %%
df_dev.head()

# %%
df_dev['clean_text'][7]

# %%
X_dev = bow.transform(df_dev['clean_text']).toarray()
X_dev

# %%
y_dev = df_dev['sentiment']

# %%
predictions_NB_dev = clf_nb.predict(X_dev)

# %%
from sklearn import metrics

print("Accuracy = ",metrics.accuracy_score(y_dev,predictions_NB_dev) * 100,'%')

# %% [markdown]
# ##### Multinomial Naive Bayes test set evaluation

# %%
#Using testet for evaluation
#Change testset parameter to testsets[0],testsets[1],testsets[2]
df_test = pd.read_csv(testsets[2],sep='\t',header=None)
df_test.columns = ['id','sentiment','text']
df_test['sentiment'] = df_test['sentiment'].map(sent_category_dict)
df_test['clean_text'] = df_test['text'].apply(preprocess_text)

X_test = bow.transform(df_test['clean_text']).toarray()
y_test = df_test['sentiment']

#CHANGE CLASSIFIER HERE
pred_t = clf_nb.predict(X_test)

#Making a df with predicitions and sentiment
df_pred_t = df_test[['id']]
df_pred_t = df_pred_t.astype(str)

#Adding sentiment column
df_pred_t = df_pred_t.assign(sentiment =  pred_t)

#Mapping the values in 'sentiment' column
dict_pos_neg = {0:'negative',1:'neutral',2:'positive'}
df_pred_t['sentiment'] = df_pred_t['sentiment'].apply(lambda x : dict_pos_neg[x])

#Before converting to dictionary we need to set 'id' column as index
df_pred_t.set_index('id',inplace=True)

dict_pred_t = df_pred_t.to_dict()['sentiment']

# %%
#Change testset parameter to testsets[0],testsets[1],testsets[2]
evaluate(dict_pred_t,testsets[2],'clf_nb')

#Change testset parameter to testsets[0],testsets[1],testsets[2]
confusion(dict_pred_t,testsets[2],'clf_nb')

# %% [markdown]
# ### TFIDF Naive Bayes

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 20000, ngram_range=(1,3))
X = tfidf.fit_transform(df_tweets['clean_text']).toarray()
y = df_tweets['sentiment']

# %%
from sklearn.naive_bayes import MultinomialNB
clf_nb = MultinomialNB()

# %%
clf_nb.fit(X,y)

# %%
df_dev = pd.read_csv('twitter-dev-data.txt',sep='\t',header=None)
df_dev.columns = ['id','sentiment','text']
df_dev['sentiment'] = df_dev['sentiment'].map(sent_category_dict)
df_dev['clean_text'] = df_dev['text'].apply(preprocess_text)
X_dev = bow.transform(df_dev['clean_text']).toarray()
y_dev = df_dev['sentiment']

# %%
predictions_NB_dev = clf_nb.predict(X_dev)

# %%
from sklearn import metrics

print("Accuracy = ",metrics.accuracy_score(y_dev,predictions_NB_dev) * 100,'%')

# %% [markdown]
# #### Test set evaluation TFIDF Naive Bayes

# %%
#Using testet for evaluation
#Change testset parameter to testsets[0],testsets[1],testsets[2]
df_test = pd.read_csv(testsets[0],sep='\t',header=None)
df_test.columns = ['id','sentiment','text']
df_test['sentiment'] = df_test['sentiment'].map(sent_category_dict)
df_test['clean_text'] = df_test['text'].apply(preprocess_text)

X_test = tfidf.transform(df_test['clean_text']).toarray()
y_test = df_test['sentiment']

#CHANGE CLASSIFIER HERE
pred_t = clf_nb.predict(X_test)

#Making a df with predicitions and sentiment
df_pred_t = df_test[['id']]
df_pred_t = df_pred_t.astype(str)

#Adding sentiment column
df_pred_t = df_pred_t.assign(sentiment =  pred_t)

#Mapping the values in 'sentiment' column
dict_pos_neg = {0:'negative',1:'neutral',2:'positive'}
df_pred_t['sentiment'] = df_pred_t['sentiment'].apply(lambda x : dict_pos_neg[x])

#Before converting to dictionary we need to set 'id' column as index
df_pred_t.set_index('id',inplace=True)

dict_pred_t = df_pred_t.to_dict()['sentiment']

#Change testset parameter to testsets[0],testsets[1],testsets[2]
evaluate(dict_pred_t,testsets[0],'clf_nb')

#Change testset parameter to testsets[0],testsets[1],testsets[2]
confusion(dict_pred_t,testsets[0],'clf_nb')

# %% [markdown]
# TFIDF for Linear SVC model

# %%
#TF-IDF Vectorization for unigram,bigram and trigram extraction
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 40000, ngram_range=(1,3))
X = tfidf.fit_transform(df_tweets['clean_text']).toarray()

# %%
y = df_tweets['sentiment']

# %%
#Initializing model
from sklearn.svm import LinearSVC

clf_lin_svm = LinearSVC(dual=False,class_weight='balanced')

# %%
clf_lin_svm.fit(X,y)

# %%
#Predicting on development data
X_dev = tfidf.transform(df_dev['clean_text']).toarray()
y_dev = df_dev['sentiment']

# %%
predictions_lin_svm = clf_lin_svm.predict(X_dev)

# %%
print("Accuracy = ",metrics.accuracy_score(y_dev,predictions_lin_svm) * 100,'%')

# %% [markdown]
# ##### Evaluation of LinearSVC on testset

# %%
#Using testet for evaluation
#Change testset parameter to testsets[0],testsets[1],testsets[2]
df_test = pd.read_csv(testsets[2],sep='\t',header=None)
df_test.columns = ['id','sentiment','text']
df_test['sentiment'] = df_test['sentiment'].map(sent_category_dict)
df_test['clean_text'] = df_test['text'].apply(preprocess_text)

X_test = tfidf.transform(df_test['clean_text']).toarray()
y_test = df_test['sentiment']

#CHANGE CLASSIFIER HERE
pred_t = clf_lin_svm.predict(X_test)

#Making a df with predicitions and sentiment
df_pred_t = df_test[['id']]
df_pred_t = df_pred_t.astype(str)

#Adding sentiment column
df_pred_t = df_pred_t.assign(sentiment =  pred_t)

#Mapping the values in 'sentiment' column
dict_pos_neg = {0:'negative',1:'neutral',2:'positive'}
df_pred_t['sentiment'] = df_pred_t['sentiment'].apply(lambda x : dict_pos_neg[x])

#Before converting to dictionary we need to set 'id' column as index
df_pred_t.set_index('id',inplace=True)

dict_pred_t = df_pred_t.to_dict()['sentiment']

#Change testset parameter to testsets[0],testsets[1],testsets[2]
evaluate(dict_pred_t,testsets[2],'clf_lin_svm')

#Change testset parameter to testsets[0],testsets[1],testsets[2]
confusion(dict_pred_t,testsets[2],'clf_lin_svm')

# %% [markdown]
# #### BOW for Linear SVC

# %%
#Bag of words, set binary=True since we are using it for NaiveBayes which is a probabilistic model
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(binary=True,max_features=40000,ngram_range=(1,3))
X = bow.fit_transform(df_tweets['clean_text']).toarray()

y = df_tweets['sentiment']

#Initializing model
from sklearn.svm import LinearSVC

clf_lin_svm = LinearSVC(dual=False,class_weight='balanced')

clf_lin_svm.fit(X,y)

#Predicting on development data
X_dev = bow.transform(df_dev['clean_text']).toarray()
y_dev = df_dev['sentiment']

predictions_lin_svm = clf_lin_svm.predict(X_dev)

print("Accuracy = ",metrics.accuracy_score(y_dev,predictions_lin_svm) * 100,'%')

#Using testet for evaluation
#Change testset parameter to testsets[0],testsets[1],testsets[2]
df_test = pd.read_csv(testsets[0],sep='\t',header=None)
df_test.columns = ['id','sentiment','text']
df_test['sentiment'] = df_test['sentiment'].map(sent_category_dict)
df_test['clean_text'] = df_test['text'].apply(preprocess_text)

X_test = bow.transform(df_test['clean_text']).toarray()
y_test = df_test['sentiment']

#CHANGE CLASSIFIER HERE
pred_t = clf_lin_svm.predict(X_test)

#Making a df with predicitions and sentiment
df_pred_t = df_test[['id']]
df_pred_t = df_pred_t.astype(str)

#Adding sentiment column
df_pred_t = df_pred_t.assign(sentiment =  pred_t)

#Mapping the values in 'sentiment' column
dict_pos_neg = {0:'negative',1:'neutral',2:'positive'}
df_pred_t['sentiment'] = df_pred_t['sentiment'].apply(lambda x : dict_pos_neg[x])

#Before converting to dictionary we need to set 'id' column as index
df_pred_t.set_index('id',inplace=True)

dict_pred_t = df_pred_t.to_dict()['sentiment']

#Change testset parameter to testsets[0],testsets[1],testsets[2]
evaluate(dict_pred_t,testsets[0],'clf_lin_svm')

#Change testset parameter to testsets[0],testsets[1],testsets[2]
confusion(dict_pred_t,testsets[0],'clf_lin_svm')

# %% [markdown]
# #### Random Forest using TFIDF

# %%
from sklearn.ensemble import RandomForestClassifier

clf_rfc = RandomForestClassifier(n_estimators = 30,class_weight='balanced',max_depth=30) 

clf_rfc.fit(X,y)

# %%
predictions_rfc_dev = clf_rfc.predict(X_dev)

# %%
from sklearn import metrics
print("Accuracy = ",metrics.accuracy_score(y_dev,predictions_rfc_dev) * 100,'%')

# %% [markdown]
# #### Random Forest Evaluation Test Set

# %%
#Using testet for evaluation
#Change testset parameter to testsets[0],testsets[1],testsets[2]
df_test = pd.read_csv(testsets[2],sep='\t',header=None)
df_test.columns = ['id','sentiment','text']
df_test['sentiment'] = df_test['sentiment'].map(sent_category_dict)
df_test['clean_text'] = df_test['text'].apply(preprocess_text)

X_test = tfidf.transform(df_test['clean_text']).toarray()
y_test = df_test['sentiment']

#CHANGE CLASSIFIER HERE
pred_t = clf_rfc.predict(X_test)

#Making a df with predicitions and sentiment
df_pred_t = df_test[['id']]
df_pred_t = df_pred_t.astype(str)

#Adding sentiment column
df_pred_t = df_pred_t.assign(sentiment =  pred_t)

#Mapping the values in 'sentiment' column
dict_pos_neg = {0:'negative',1:'neutral',2:'positive'}
df_pred_t['sentiment'] = df_pred_t['sentiment'].apply(lambda x : dict_pos_neg[x])

#Before converting to dictionary we need to set 'id' column as index
df_pred_t.set_index('id',inplace=True)

dict_pred_t = df_pred_t.to_dict()['sentiment']


evaluate(dict_pred_t,testsets[2],'clf_rfc')

#Confusion matrix for testset
confusion(dict_pred_t,testsets[2],'clf_rfc')




