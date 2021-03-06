# *_Sentimental analysis on GME tweets and relation to GME value_*

<img src="https://github.com/hempelc/sentimental_analysis_GME_twitter/blob/main/image.png" alt="GME" width="400"/>

This notebook tests if the emotions of GameStop stock (GME)-related tweets were correlated to the GME value during the GME boom on 28th January 2021.

Therefore, we train a model to classify tweets into "positive" and "negative" based on emotions. <br/> Then we scrape tweets from the 28th January 2021 that contain GME-relevant hashtags. <br/>  Finally, we obtain GME stock market data from 28th January 2021 and graph it against GME tweet emotions.

The model training follows the sentimental analysis tutorial posted here: https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk

Note: ***this notebook is broken***! Extracting old stock market data is not free - at the point in time when I generated this notebook, I was not able to access the relevant stock market data anymore!

# Required modules


```python
import re, string, random, nltk, json, glob, numpy as np, pandas as pd, yfinance as yf, matplotlib.pyplot as plt

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from datetime import datetime
```

# 1. Make a model to classify tweets into "positive" and "negative" emotions

## 1.1 Download  example and denoising data


```python
nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
```

    [nltk_data] Downloading package twitter_samples to
    [nltk_data]     /Users/christopherhempel/nltk_data...
    [nltk_data]   Package twitter_samples is already up-to-date!
    [nltk_data] Downloading package punkt to
    [nltk_data]     /Users/christopherhempel/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     /Users/christopherhempel/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     /Users/christopherhempel/nltk_data...
    [nltk_data]   Package averaged_perceptron_tagger is already up-to-
    [nltk_data]       date!
    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/christopherhempel/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True



## 1.2 Define functions


```python
# Remove noise from tweets
def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# Get tweets from list for model
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)
```

## 1.3 Load in positive and negative tweet examples and clean data


```python
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

stop_words = stopwords.words('english')

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset
```

## 1.4 Train Naive Bayes classifier model


```python
random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]

classifier = NaiveBayesClassifier.train(train_data)

print("Naive Bayes classifier model accuracy is:", classify.accuracy(classifier, test_data))

print(classifier.show_most_informative_features(10))
```

    Naive Bayes classifier model accuracy is: 0.9946666666666667
    Most Informative Features
                          :( = True           Negati : Positi =   2078.6 : 1.0
                    follower = True           Positi : Negati =     34.6 : 1.0
                         sad = True           Negati : Positi =     33.4 : 1.0
                         bam = True           Positi : Negati =     21.4 : 1.0
                        blog = True           Positi : Negati =     14.8 : 1.0
                      arrive = True           Positi : Negati =     13.1 : 1.0
                        glad = True           Positi : Negati =     12.5 : 1.0
                     welcome = True           Positi : Negati =     12.2 : 1.0
                     awesome = True           Positi : Negati =     12.1 : 1.0
                        kill = True           Negati : Positi =     11.8 : 1.0
    None


# 1.5 Test classification on custom example tweet


```python
custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."

custom_tokens = remove_noise(word_tokenize(custom_tweet))

print("Custom tweet: ", custom_tweet)
print("Classification: ", classifier.classify(dict([token, True] for token in custom_tokens)))
```

    Custom tweet:  I ordered just once from TerribleCo, they screwed up, never used the app again.
    Classification:  Negative


# 2. Scrape and classify GME tweets from 28th January 2021

## 2.1 Use APIFY (web scraping platform) to scrape tweets using GME-relevant hashtags into JSON files and save JSON files to working directory

Note: we did the scraping and the resulting JSON files are saved to the directory containing this Jupyter Notebook.
We scraped for tweets from 28th January 2021 containing the hashtags "gamestop", "gme", "robinhood", "shortsqueeze", and "wallstreetbets".

## 2.2 Import and clean scraped tweets


```python
json_files = glob.glob("*.json")
twitter_files = np.zeros(len(json_files))

tweetsdict = [dict() for x in range(len(json_files))]

i = 0
for file in json_files:
  with open(file) as f:
    tweets = json.load(f)
    tweetsdict[i] = tweets
    i += 1

gme_tweets = []
for tweets in tweetsdict:
  for tweet in tweets:
    gme_tweets.append(tweet['full_text'])

print("Total number of tweets: ", len(gme_tweets))
```

    Total number of tweets:  1954


## 2.3 Classify tweets and define ratio positive:negative tweets


```python
tweet_classification=[]
for tweet in gme_tweets:
  tokens=remove_noise(word_tokenize(tweet))
  tweet_classification.append(classifier.classify(dict([token, True] for token in tokens)))

ratio_pos_neg=tweet_classification.count('Positive')/tweet_classification.count('Negative')

print(f"Ratio positive:negative tweets: {ratio_pos_neg}")
```

    Ratio positive:negative tweets: 1.1543550165380374


# 3. Obtain GME stock market data from 28th January 2021 and graph it against GME tweet emotions

## 3.1 Download GME stock data

Note: the selected time frame is not the time frame in which the GME boom occurred - this data was not freely available anymore.


```python
# Define the ticker symbol
tickerSymbol = 'GME'

# Get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# Get historical prices for this ticker
tickerDf = tickerData.history(period='1d', interval='15m', start='2022-01-10', end='2022-01-30')

#see your data
tickerDf.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividends</th>
      <th>Stock Splits</th>
    </tr>
    <tr>
      <th>Datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-10 09:30:00-05:00</th>
      <td>134.990005</td>
      <td>135.690002</td>
      <td>126.500000</td>
      <td>127.643402</td>
      <td>755828</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-01-10 09:45:00-05:00</th>
      <td>127.550003</td>
      <td>129.470001</td>
      <td>126.822800</td>
      <td>127.639999</td>
      <td>350949</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-01-10 10:00:00-05:00</th>
      <td>127.510002</td>
      <td>128.249893</td>
      <td>120.000099</td>
      <td>121.580002</td>
      <td>680539</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-01-10 10:15:00-05:00</th>
      <td>121.889999</td>
      <td>125.199898</td>
      <td>121.440002</td>
      <td>124.260101</td>
      <td>461321</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-01-10 10:30:00-05:00</th>
      <td>124.269997</td>
      <td>124.269997</td>
      <td>120.300003</td>
      <td>120.839104</td>
      <td>321064</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 3.2 Simulate volume of stock dataset

Note: we simulate stock dataset volume for the GME boom time frame.


```python
time = pd.date_range("2021-01-27", periods=1200, freq="15T")
random_volume = random.sample(range(1000000, 9999999), 1200)
volume_df = pd.DataFrame({'time':time, 'volume':random_volume}).set_index('time')
volume_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>volume</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-01-27 00:00:00</th>
      <td>3215888</td>
    </tr>
    <tr>
      <th>2021-01-27 00:15:00</th>
      <td>8463827</td>
    </tr>
    <tr>
      <th>2021-01-27 00:30:00</th>
      <td>6587678</td>
    </tr>
    <tr>
      <th>2021-01-27 00:45:00</th>
      <td>9132865</td>
    </tr>
    <tr>
      <th>2021-01-27 01:00:00</th>
      <td>4832439</td>
    </tr>
  </tbody>
</table>
</div>



## 3.3 Process tweet date and time

Note: this needs to be polished to accurately reflect times of the web-scraped tweets.


```python
# Get times per tweet and make dataframe with times and classifications

tweet_times = []
for tweets in tweetsdict:
  for tweet in tweets:
    time=datetime.strptime(tweet['created_at'][2:19].replace("T", " "), '%y-%m-%d %H:%M:%S')
    tweet_times.append(time)

tweet_df = pd.DataFrame({'time':tweet_times, 'class':tweet_classification}).set_index('time')

tweet_df_counts = pd.get_dummies(tweet_df, columns=['class']).resample('15T').sum()

tweet_df_counts_clean = tweet_df_counts[(tweet_df_counts.T != 0).any()]

tweet_df_counts_clean.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class_Negative</th>
      <th>class_Positive</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-01-27 19:45:00</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2021-01-27 22:45:00</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2021-01-28 00:15:00</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2021-01-28 01:15:00</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2021-01-28 02:15:00</th>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 3.4 Merge stock data and tweet data


```python
mergedDf = tweet_df_counts_clean.merge(volume_df, left_index=True, right_index=True).reset_index()
mergedDf.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>class_Negative</th>
      <th>class_Positive</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-27 19:45:00</td>
      <td>1</td>
      <td>0</td>
      <td>4688632</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-27 22:45:00</td>
      <td>1</td>
      <td>0</td>
      <td>4097852</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-28 00:15:00</td>
      <td>1</td>
      <td>0</td>
      <td>3240474</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-28 01:15:00</td>
      <td>1</td>
      <td>0</td>
      <td>3391354</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-28 02:15:00</td>
      <td>1</td>
      <td>1</td>
      <td>1011225</td>
    </tr>
  </tbody>
</table>
</div>



## 3.5 Plot volume against positive and negative tweets

Note: this graph is made from simulated data, so it doesn't make sense.


```python
fig, ax1 = plt.subplots()

ax1.set_xlabel('Date')
ax1.set_ylabel('# positive/negative')
ax1.plot(mergedDf["time"], mergedDf["class_Negative"], color='tab:red')
ax1.plot(mergedDf["time"], mergedDf["class_Positive"], color='tab:blue')

ax2 = ax1.twinx() # instantiate a second axis that shares the same x-axis

ax2.set_ylabel('Volume')
ax2.plot(mergedDf["time"], mergedDf["volume"], color='tab:green')

fig.tight_layout() # otherwise the right y-label is slightly clipped
fig.autofmt_xdate() # automatically rotates x axis labels
fig.savefig("figure.png", facecolor='white')
```



![png](https://github.com/hempelc/sentimental_analysis_GME_twitter/blob/main/figure.png)    
