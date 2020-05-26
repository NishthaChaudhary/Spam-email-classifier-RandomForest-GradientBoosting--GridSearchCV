# Read Text
import pandas as pd
data = pd.read_csv("C:/Users/nisht/Anaconda3_n/envs/LL/NLP/Ch01/01_10/End/SMSSpamCollection.tsv", sep='\t')
data.columns = ['label', 'body_text']

# Create feature for text message length
data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))
data.head()

# Create feature for % of text that is punctuation
import string
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100
data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))
data.head()

# Evaluate created features
from matplotlib import pyplot
import numpy as np
%matplotlib inline

bins = np.linspace(0, 200, 40)
pyplot.hist(data[data['label']=='spam']['body_len'], bins, alpha=0.5, label='spam')
pyplot.hist(data[data['label']=='ham']['body_len'], bins, alpha=0.5, label='ham')
pyplot.legend(loc='upper left')
pyplot.show()

bins = np.linspace(0, 50, 40)
pyplot.hist(data[data['label']=='spam']['punct%'], bins, alpha=0.5, label='spam')
pyplot.hist(data[data['label']=='ham']['punct%'], bins, alpha=0.5, label='ham')
pyplot.legend(loc='upper right')
pyplot.show()

bins = np.linspace(0, 200, 40)
pyplot.hist(data['body_len'], bins)
pyplot.title("Body Length Distribution")
pyplot.show()

bins = np.linspace(0, 50, 40)
pyplot.hist(data['punct%'], bins)
pyplot.title("Punctuation % Distribution")
pyplot.show()

# Box-Cox Transformation
for i in [1,2,3,4,5]:
    pyplot.hist((data['punct%'])**(1/i), bins=40)
    pyplot.title("Transformation: 1/{}".format(str(i)))
    pyplot.show()
