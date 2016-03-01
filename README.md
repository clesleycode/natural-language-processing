Intro to Natural Language Processing
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958) and [ADI](https://adicu.com)

## Table of Contents

- [0.0 Setup](#00-setup)
    + [0.1 Python and Anaconda](#01-python--anaconda)
    + [0.2 Libraries](#02-libraries)
    + [0.3 Other](#03-other)
- [1.0 Background](#10-background)
    + [1.1 What is NLP?](#11-what-is-nlp)
    + [1.2 Why is NLP Important?](#12-why-is-nlp-importance)
    + [1.3 Why is NLP a "hard" problem?](#13-why-is-nlp-a-hard-problem)
- [2.0 Regular Expressions](#20-regular-expressions)
    + [2.1 Simplest Form](#21-simplest-form)
    + [2.2 Case Sensitivity](#22-case-sensitivity)
    + [2.3 Disjunctions](#23-disjunctions) 
    + [2.4 Ranges](#24-ranges) 
    + [2.5 Exclusions](#25-exclusions) 
    + [2.6 Question Marks](#26-question-marks) 
    + [2.7 Kleene Star](#27-kleene-star) 
    + [2.8 Wildcards](#28-wildcards) 
    + [2.9 Kleene+](#29-kleene) 
- [3.0 Word Tagging and Models](#30-word-tagging--models)
    + [3.1 NLTK Parts of Speech Tagger](#31-nltk-parts-of-speech-tagger)
        * [3.1.1 Ambiguity](#311-ambiguity)
    + [3.2 Unigram Models](#32-unigram-models)
    + [3.3 Bigram Models](#33-bigram-models)
- [4.0 Normalizing Text](#40-normalizing-text)
    + [4.1 Stemming](#41-stemming)
        * [4.1.1 What is Stemming?](#411-what-is-stemming)
        * [4.1.2 Types of Stemmers](#412-types-of-stemmers)
    + [4.2 Lemmatization](#42-lemmatization)
        * [4.2.1 What is Lemmatization?](#421-what-is-lemmatization)
        * [4.2.2 WordNetLemmatizer?](#422-wordnetlemmatizer)
    + [4.3 ](#43-)
- [5.0 Sentiment Analysis](#50-sentiment-analysis)
    + [5.1 Preparing the Data](#51-preparing-the-data)
        * [5.1.1 Training Data](#511-training-data)
        * [5.1.2 Test Data](#512-test-data)
    + [5.2 Building a Classifier](#52-building-a-classifier)
    + [5.3 Classification](#53-classification)


## 0.0 Setup

This guide was written in Python 2.7.

### 0.1 Python & Anaconda

Download [Python](https://www.python.org/downloads/) and [Anaconda](http://docs.continuum.io/anaconda/install). 

### 0.2 Libraries

We'll be working with the re library for regular expressions and nltk for natural language processing techniques, so make sure to install them! Lastly, we'll be doing a few visualizations in matplotlib. To install these libraries, enter the following commands into your terminal: 

``` 
conda install nltk
conda install re
conda install matplotlib
```

### 0.3 Other

Since we'll be working on textual analysis, we'll be using datasets that are already well established and widely used. To gain access to these datasets, enter the following command into your command line: 
```
sudo python -m nltk.downloader all
```

Now you're all set to begin!

## 1.0 Background


### 1.1 What is NLP? 

Natural Language Processing, or NLP, is an area of computer science that focuses on developing techniques to produce machine-driven analyses of text.

### 1.2 Why is Natural Language Processing Important? 

NLP expands the sheer amount of data that can be used for insight. Since so much of the data we have available is in the form of text, this is extremely important to data science!

A specific common application of NLP is each time you use a language conversion tool. The techniques used to accurately convert text from one language to another very much falls under the umbrella of "natural language processing."

### 1.3 Why is NLP a "hard" problem? 

Language is inherently ambiguous. Once person's interpretation of a sentence may very well differ from another person's interpretation. Because of this inability to consistently be clear, there are no perfect NLP techniques. 


## 2.0 Regular Expressions

A regular expression is a sequence of characters that define a string.

### 2.1 Simplest Form

The simplest form of a regular expression is a sequence of characters contained within <b>two backslashes</b>. For example, <i>python</i> would be  

``` 
\python
```

### 2.2 Case Sensitivity

Regular Expressions are <b>case sensitive</b>, which means 

``` 
\p and \P
```
are distinguishable from eachother. This means <i>python</i> and <i>Python</i> would have to be represented differently, as follows: 

``` 
\python and \Python
```

### 2.3 Disjunctions

If you want a regular expression to represent both <i>python</i> and <i>Python</i>, however, you can use <b>brackets</b> or the <b>pipe</b> symbol as the disjunction of the two forms. For example, 
``` 
[Pp]ython or \Python|python
```
could represent either <i>python</i> or <i>Python</i>. Likewise, 

``` 
[0123456789]
```
would represent a single integer digit. The pipe symbols are typically used for interchangable strings, such as in the following example:

```
\dog|cat
```

### 2.4 Ranges

If we want a regular expression to express the disjunction of a range of characters, we can use a <b>dash</b>. For example, instead of the previous example, we can write 

``` 
[0-9]
```
Similarly, we can represent all characters of the alphabet with 

``` 
[a-z]
```

### 2.5 Exclusions

Brackets can also be used to represent what an expression <b>cannot</b> be if you combine it with the <b>caret</b> sign. For example, the expression 

``` 
[^p]
```
represents any character, special characters included, but p.

### 2.6 Question Marks 

Question marks can be used to represent the expressions containing zero or one instances of the previous character. For example, 

``` 
<i>\colou?r
```
represents either <i>color</i> or <i>colour</i>. Question marks are often used in cases of plurality. For example, 

``` 
<i>\computers?
```
can be either <i>computers</i> or <i>computer</i>. If you want to extend this to more than one character, you can put the simple sequence within parenthesis, like this:

```
\Feb(ruary)?
```
This would evaluate to either <i>February</i> or <i>Feb</i>.

### 2.7 Kleene Star

To represent the expressions containing zero or <b>more</b> instances of the previous character, we use an <b>asterisk</b> as the kleene star. To represent the set of strings containing <i>a, ab, abb, abbb, ...</i>, the following regular expression would be used:  
```
\ab*
```

### 2.8 Wildcards

Wildcards are used to represent the possibility of any character and symbolized with a <b>period</b>. For example, 

```
\beg.n
```
From this regular expression, the strings <i>begun, begin, began,</i> etc., can be generated. 

### 2.9 Kleene+

To represent the expressions containing at <b>least</b> one or more instances of the previous character, we use a <b>plus</b> sign. To represent the set of strings containing <i>ab, abb, abbb, ...</i>, the following regular expression would be used:  

```
\ab+
```

## 3.0 Word Tagging & Models

Given any sentence, you can classify each word as a noun, verb, conjunction, or any other class of words. When there are hundreds of thousands of sentences, even millions, this is obviously a large and tedious task. But it's not one that can't be solved computationally. 


### 3.1 NLTK Parts of Speech Tagger

NLTK is a package in python that provides libraries for different text processing techniques, such as classification, tokenization, stemming, parsing, but important to this example, tagging. 

``` python
import nltk 

text = nltk.word_tokenize("Python is an awesome language!")
nltk.pos_tag(text)
```

```python
[('Python', 'NNP'), ('is', 'VBZ'), ('an', 'DT'), ('awesome', 'JJ'), ('language', 'NN'), ('!', '.')]
```

Not sure what DT, JJ, or any other tag is? Just try this in your python terminal: 

```python
nltk.help.upenn_tagset('JJ')
```
``` 
JJ: adjective or numeral, ordinal
third ill-mannered pre-war regrettable oiled calamitous first separable
ectoplasmic battery-powered participatory fourth still-to-be-named
multilingual multi-disciplinary ...
```


#### 3.1.1 Ambiguity

But what if a word can be tagged as more than one part of speech? For example, the word "sink." Depending on the content of the sentence, it could either be a noun or a verb.


### 3.2 Unigram Models

```python
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
unigram_tagger.tag(brown_sents[2007])

```

```
[('Various', 'JJ'), ('of', 'IN'), ('the', 'AT'), ('apartments', 'NNS'), ('are', 'BER'), ('of', 'IN'), ('the', 'AT'), ('terrace', 'NN'), ('type', 'NN'), (',', ','), ('being', 'BEG'), ('on', 'IN'), ('the', 'AT'), ('ground', 'NN'), ('floor', 'NN'), ('so', 'QL'), ('that', 'CS'), ('entrance', 'NN'), ('is', 'BEZ'), ('direct', 'JJ'), ('.', '.')]
```

### 3.3 Bigram Models


``` python
bigram_tagger = nltk.BigramTagger(brown_tagged_sents)
bigram_tagger.tag(brown_sents[2007])
```

Notice the changes from the last time we tagged the words of this same sentence: 

```
[('Various', 'JJ'), ('of', 'IN'), ('the', 'AT'), ('apartments', 'NNS'), ('are', 'BER'), ('of', 'IN'), ('the', 'AT'), ('terrace', 'NN'), ('type', 'NN'), (',', ','), ('being', 'BEG'), ('on', 'IN'), ('the', 'AT'), ('ground', 'NN'), ('floor', 'NN'), ('so', 'CS'), ('that', 'CS'), ('entrance', 'NN'), ('is', 'BEZ'), ('direct', 'JJ'), ('.', '.')]
```

### 3.4 


## 4.0 Normalizing Text

The best data is data that's consistent - textual data usually isn't. But we can make it that way by normalizing it. To do this, we can do a number of things. 

At the very least, we can make all the text so that it's all in lowercase. You may have already done this before: 

Given a piece of text, 

``` python
raw = "OMG, Natural Language Processing is SO cool and I'm really enjoying this workshop!"
tokens = nltk.word_tokenize(raw)
tokens = [i.lower() for i in tokens]
```

```
['omg', ',', 'natural', 'language', 'processing', 'is', 'so', 'cool', 'and', 'i', "'m", 'really', 'enjoying', 'this', 'workshop', '!']
```


### 4.1 Stemming

But we can do more! 

#### 4.1.1 What is Stemming?

Stemming is the process of converting the words of a sentence to its non-changing portions. In the example of amusing, amusement, and amused above, the stem would be amus.

#### 4.1.2 Types of Stemmers

You're probably wondering how do I convert a series of words to its stems. Luckily, NLTK has a few built-in and established stemmers available for you to use! They work slightly differently since they follow different rules - which you use depends on whatever you happen to be working on. 

First, let's try the Lancaster Stemmer: 

``` python
lancaster = nltk.LancasterStemmer()
stems = [lancaster.stem(i) for i in tokens]
```

This should have the output: 
```
['omg', ',', 'nat', 'langu', 'process', 'is', 'so', 'cool', 'and', 'i', "'m", 'real', 'enjoy', 'thi', 'workshop', '!']
```

Secondly, we try the Porter Stemmer:

``` python
porter = nltk.PorterStemmer()
stem = [porter.stem(i) for i in tokens]
```

Notice how "natural" maps to "natur" instead of "nat" and "really" maps to "realli" instead of "real" in the last stemmer. 
```
['omg', ',', 'natur', 'languag', 'process', 'is', 'so', 'cool', 'and', 'i', "'m", 'realli', 'enjoy', 'thi', 'workshop', '!']
```


### 4.2 Lemmatization

#### 4.2.1 What is Lemmatization?

Lemmatization is the process of converting the words of a sentence to its dictionary form. For example, given the words amusement, amusing, and amused, the lemma for each and all would be amuse.

#### 4.2.2 WordNetLemmatizer

Once again, NLTK is awesome and has a built in lemmatizer for us to use: 

``` python
from nltk import WordNetLemmatizer

lemma = nltk.WordNetLemmatizer()
text = "Women in technology are amazing at coding"
ex = [i.lower() for i in text.split()]

lemmas = [lemma.lemmatize(i) for i in ex]
```

``` 
['woman', 'in', 'technology', 'are', 'amazing', 'at', 'coding']
```


## 5.0 Sentiment Analysis  

Before now, we've ran a few lines of code to play around with some textual data. But now we'll write some code to perform sentiment analysis on some tweets. 

So you might be asking, what exactly is "sentiment analysis"? 

Well, sentiment analysis involves building a system to collect and determine the emotional tone behind words. This is important because it allows you to gain an understanding of the attitudes, opinions and emotions of the people in your data. 

At a high level, sentiment analysis involves Natural language processing and artificial intelligence by taking the actual text element, transforming it into a format that machine can read, and using statistics to determine the actual sentiment.

### 5.1 Preparing the Data 

To accomplish sentiment analysis computationally, we have to use techniques that will allow us to learn from data that's already been labeled. 

So what's the first step? Splitting labeled data we have into two pieces, one that can "train" data and the other to give us insight on how well our model is performing. 

``` python
return all_tweets
```

#### 5.1.1 Training Data


``` python
def train_data(all_tweets, parse):
    train = []
    for i in range(parse):
        train.append(all_tweets[i])
    return train
```

#### 5.1.2 Test Data

``` python
def test_data(all_tweets, parse):
    test = []
    for i in range(parse, len(all_tweets)):
        test.append(all_tweets[i])
    return test
```

### 5.2 Building a Classifier

#### 5.2.1 Features

```python
def get_words(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words
```

``` python
def get_features(words):
    words = nltk.FreqDist(words)
    features = words.keys()
    return features
```

#### 5.2.2 Feature Extraction

``` python
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
```

``` python
training_set = nltk.classify.apply_features(extract_features, tweets)
```

#### 5.2.3 Training the Classifier

``` python
classifier = nltk.NaiveBayesClassifier.train(training_set)
```

### 5.3 Classification

```python

example = "this workshop is awesome."

print classifier.classify(extract_features(example.split()))

```




