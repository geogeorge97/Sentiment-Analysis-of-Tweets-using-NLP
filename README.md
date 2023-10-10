# Sentiment Analysis of Tweets using Python and NLP

Designed and implemented a sentiment classifier to classify tweets as positive, negative or neutral utilizing
Linear Support Vector Classifier (SVC); Obtained accuracy of 63% and f1-score of 60%. Tweets initially
cleaned using regular expressions to remove hyperlinks, URLs and unnecessary characters. TweetTokenizer
utilized as it handles emojis, hashtags and other elements of a tweet well. As sentiment of tweet is the only
point of interest, stemming is done instead of lemmatization

### EDA
• Firstly, we load in the training data ‘twitter-training-data.txt’ into a dataframe using pandas.
This enables easy manipulation and greater interpretability. Then we move on to explore the
data (EDA). Here, we observe that the target class i.e sentiment is imbalanced and has
higher number of ‘neutral’ values than ‘positive’ or ‘negative’.
• Although techniques like SMOTE() can be used to oversample and solve the class imbalance
we do not use that approach here, but keep this in mind while doing our predictions.
• Next, we look at the length of the tweets by making a new column for the same. We can see
that there are some outliers that are either too long or too short.
Text pre-processing
• We define a function to pre-process the tweets. Each tweet is first lowered. Reusing regex
code from assignment 1, we remove urls that start with http or https or www.
• It is noticed that there are a few html reference characters of the form ‘&amp;’. These are
removed as well.
• @ mentions are removed with the text accompanying it as these are usually names and
don’t give us much semantic information.
• Except for the punctuation used in emojis like (): etc. all other non-alphanumeric and
punctuation is removed.
• We use a special tokenizer called the TweetTokenizer to tokenize our texts. This handles
emojis, hashtags and other elements of a tweet well.
• We also remove all stopwords and perform stemming using the SnowballStemmer, which is
an enhanced version of the standard porter stemmer. We use stemming instead of
lemmatization because we are not really interested in meaningful words here as this is a
sentiment analysis question. We merely want to find out the sentiment of the tweet, so the
words need not make sense.

### Classifiers

#### Multinomial Naïve Bayes
• Although TF-IDF is also a viable option, since multinomial naïve bayes is a probabilistic
model, it is better to use bag of words model (BOW) instead.
• For BOW we extract unigram, bigram and trigram features using the ngram_range
parameter in the CountVectorizer. We also set binary=True as it is recommended when
using probabilistic models like Naïve Bayes, max_features is set to 20000.
• We then fit the model and test our predicitions on the development data after applying
the same preprocessing function on it, use bow.transform to get the word vectors here.
• We obtain an accuracy of 62.4 % on our development data.
• Macro-average f1 scores for 3 testsets:
We can see consistent performance across all 3 test sets.
On the other hand when we use TFIDF for Naïve Bayes using the same number of
features i.e 20000 and n-grams, we get much lower f1 scores. Testset 1 f1 score is shown
below.

#### Linear SVC model
• We use TFIDF instead of bag of words when it comes to the svm model (LinearSVC). TF-
IDF model contains information on the more important words and the less important
ones as well and considers the frequency of the word in the entire corpus. This makes it
particularly useful in ML models.
• We intialize tfidf vectorizer with max_features = 40,000 and ngram_range = (1,3) to
generate extract unigram, bigram and trigram features.
• Here, n_samples > n_features so we set the dual parameter = False. Also, class_weights
are automatically balanced used the class_weight=’balanced’ parameter. The “balanced”
mode uses the values of y to automatically adjust weights inversely proportional to class
frequencies in the input data
• On the development dataset we obtain an accuracy of 62.649 %
• Evaluating on testsets we obtain f1 scores as –
Random Forest model
• Same as above we utilize TFIDF vectors for our random forest model. We initialize the
random forest with 30 trees, max tree depth as 30 and class_weight = ‘balanced’. The
class_weight parameter helps to take care of the class imbalance.
• Accuracy on development set = 57.65 %
• Macro averaged f1 score on testsets = 0.473, 0.503, 0.452.
It is clear that Multinomial Naïve Bayes and LinearSVC are the better models for this
scenario
