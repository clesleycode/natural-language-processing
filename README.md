Intro to Natural Language Processing
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958) and [ADI](https://adicu.com). Last major update was in 2017 and isn't being actively maintained. 

## Table of Contents

- [0.0 Setup](#00-setup)
    + [0.1 Python & Pip](#01-python--pip)
    + [0.2 Libraries](#02-libraries)
    + [0.3 Other](#03-other)
- [1.0 Background](#10-background)
    + [1.1 What is NLP?](#11-what-is-nlp)
    + [1.2 Why is NLP Important?](#12-why-is-nlp-importance)
    + [1.3 Why is NLP a "hard" problem?](#13-why-is-nlp-a-hard-problem)
    + [1.4 Glossary](#14-glossary)
- [2.0 Sentiment Analysis](#20-sentiment-analysis)
    + [2.1 Preparing the Data](#21-preparing-the-data)
        * [2.1.1 Training Data](#211-training-data)
        * [2.1.2 Test Data](#212-test-data)
    + [2.2 Building a Classifier](#22-building-a-classifier)
    + [2.3 Classification](#53-classification)
    + [2.4 Accuracy](#24-accuracy)
- [3.0 Regular Expressions](#30-regular-expressions)
    + [3.1 Simplest Form](#31-simplest-form)
    + [3.2 Case Sensitivity](#32-case-sensitivity)
    + [3.3 Disjunctions](#33-disjunctions) 
    + [3.4 Ranges](#34-ranges) 
    + [3.5 Exclusions](#35-exclusions) 
    + [3.6 Question Marks](#36-question-marks) 
    + [3.7 Kleene Star](#37-kleene-star) 
    + [3.8 Wildcards](#38-wildcards) 
    + [3.9 Kleene+](#39-kleene) 
- [4.0 Word Tagging and Models](#40-word-tagging-and-models)
    + [4.1 NLTK Parts of Speech Tagger](#41-nltk-parts-of-speech-tagger)
        * [4.1.1 Ambiguity](#411-ambiguity)
    + [4.2 Unigram Models](#42-unigram-models)
    + [4.3 Bigram Models](#43-bigram-models)
- [5.0 Normalizing Text](#40-normalizing-text)
    + [5.1 Stemming](#51-stemming)
        * [5.1.1 What is Stemming?](#511-what-is-stemming)
        * [5.1.2 Types of Stemmers](#512-types-of-stemmers)
    + [5.2 Lemmatization](#52-lemmatization)
        * [5.2.1 What is Lemmatization?](#521-what-is-lemmatization)
        * [5.2.2 WordNetLemmatizer?](#522-wordnetlemmatizer)
- [6.0 Final Words](#60-final-words)
    + [6.1 Resources](#61-resources)
    + [6.2 More Stuff](#62-mini-courses)


## 0.0 Setup

This guide was written in Python 3.6.

### 0.1 Python & Anaconda

Download [Python](https://www.python.org/downloads/) and [Pip](http://docs.continuum.io/anaconda/install).


### 0.2 Libraries

We'll be working with the `re` library for regular expressions and nltk for natural language processing techniques, so make sure to install them! To install these libraries, enter the following commands into your terminal: 

``` 
pip3 install re
pip3 install nltk
```

### 0.3 Other

Since we'll be working on textual analysis, we'll be using datasets that are already well established and widely used. To gain access to these datasets, enter the following command into your command line: (Note that this might take a few minutes!)

```
sudo python3 -m nltk.downloader all
```

Lastly, download the data we'll be working with in this example! 

[Positive Tweets](https://github.com/lesley2958/natural-language-processing/blob/master/pos_tweets.txt) <br>
[Negative Tweets](https://github.com/lesley2958/natural-language-processing/blob/master/neg_tweets.txt)

Now you're all set to begin!

## 1.0 Background


### 1.1 What is NLP? 

Natural Language Processing, or NLP, is an area of computer science that focuses on developing techniques to produce machine-driven analyses of text.

### 1.2 Why is Natural Language Processing Important? 

NLP expands the sheer amount of data that can be used for insight. Since so much of the data we have available is in the form of text, this is extremely important to data science!

A specific common application of NLP is each time you use a language conversion tool. The techniques used to accurately convert text from one language to another very much falls under the umbrella of "natural language processing."

### 1.3 Why is NLP a "hard" problem? 

Language is inherently ambiguous. Once person's interpretation of a sentence may very well differ from another person's interpretation. Because of this inability to consistently be clear, it's hard to have an NLP technique that works perfectly. 

### 1.4 Glossary

Here is some common terminology that we'll encounter throughout the workshop:

<b>Corpus: </b> (Plural: Corpora) a collection of written texts that serve as our datasets.

<b>nltk: </b> (Natural Language Toolkit) the python module we'll be using repeatedly; it has a lot of useful built-in NLP techniques.

<b>Token: </b> a string of contiguous characters between two spaces, or between a space and punctuation marks. A token can also be an integer, real, or a number with a colon.


## 2.0 Sentiment Analysis  

So you might be asking, what exactly is "sentiment analysis"? 

Well, sentiment analysis involves building a system to collect and determine the emotional tone behind words. This is important because it allows you to gain an understanding of the attitudes, opinions and emotions of the people in your data. 

At a high level, sentiment analysis involves Natural language processing and artificial intelligence by taking the actual text element, transforming it into a format that a machine can read, and using statistics to determine the actual sentiment.

### 2.1 Preparing the Data 

To accomplish sentiment analysis computationally, we have to use techniques that will allow us to learn from data that's already been labeled. 

So what's the first step? Formatting the data so that we can actually apply NLP techniques. 

``` python
import nltk

def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})
```

Here, `format_sentence` changes a piece of text, in this case a tweet, into a dictionary of words mapped to True booleans. Though not obvious from this function alone, this will eventually allow us to train  our prediction model by splitting the text into its tokens, i.e. <i>tokenizing</i> the text.

``` 
{'!': True, 'animals': True, 'are': True, 'the': True, 'ever': True, 'Dogs': True, 'best': True}
```

You'll learn about why this format is important is section 2.2.

Using the data on the github repo, we'll actually format the positively and negatively labeled data.

``` python
pos = []
with open("./pos_tweets.txt") as f:
    for i in f: 
        pos.append([format_sentence(i), 'pos'])
```

``` python
neg = []
with open("./neg_tweets.txt") as f:
    for i in f: 
        neg.append([format_sentence(i), 'neg'])
```


#### 2.1.1 Training Data

Next, we'll split the labeled data we have into two pieces, one that can "train" data and the other to give us insight on how well our model is performing. The training data will inform our model on which features are most important.

``` python
training = pos[:int((.9)*len(pos))] + neg[:int((.9)*len(neg))]
```

#### 2.1.2 Test Data

We won't use the test data until the very end of this section, but nevertheless, we save the last 10% of the data to check the accuracy of our model. 

``` python
test = pos[int((.1)*len(pos)):] + neg[int((.1)*len(neg)):]
```

### 2.2 Building a Classifier

All NLTK classifiers work with feature structures, which can be simple dictionaries mapping a feature name to a feature value. In this example, we’ve used a simple bag of words model where every word is a feature name with a value of True.

``` python
from nltk.classify import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(training)
```
 
To see which features informed our model the most, we can run this line of code:

```python
classifier.show_most_informative_features()
```

```
Most Informative Features
        no = True                neg : pos    =     20.6 : 1.0
    awesome = True               pos : neg    =     18.7 : 1.0
    headache = True              neg : pos    =     18.0 : 1.0
   beautiful = True              pos : neg    =     14.2 : 1.0
        love = True              pos : neg    =     14.2 : 1.0
          Hi = True              pos : neg    =     12.7 : 1.0
        glad = True              pos : neg    =      9.7 : 1.0
       Thank = True              pos : neg    =      9.7 : 1.0
         fan = True              pos : neg    =      9.7 : 1.0
        lost = True              neg : pos    =      9.3 : 1.0
```

### 2.3 Classification

Just to see that our model works, let's try the classifier out with a positive example: 

```python
example1 = "this workshop is awesome."

print(classifier.classify(format_sentence(example1)))
```

```
'pos'
```

Now for a negative example:

``` python
example2 = "this workshop is awful."

print(classifier.classify(format_sentence(example2)))
```

```
'neg'
```

### 2.4 Accuracy

Now, there's no point in building a model if it doesn't work well. Luckily, once again, nltk comes to the rescue with a built in feature that allows us find the accuracy of our model.

``` python
from nltk.classify.util import accuracy
print(accuracy(classifier, test))
```

``` 
0.9562326869806094
```

Turns out it works decently well!

But it could be better! I think we can agree that the data is kind of messy - there are typos, abbreviations, grammatical errors of all sorts... So how do we handle that? Can we handle that? 


## 3.0 Regular Expressions

A regular expression is a sequence of characters that define a string.

### 3.1 Simplest Form

The simplest form of a regular expression is a sequence of characters contained within <b>two backslashes</b>. For example, <i>python</i> would be  

``` 
\python
```

### 3.2 Case Sensitivity

Regular Expressions are <b>case sensitive</b>, which means 

``` 
\p and \P
```
are distinguishable from eachother. This means <i>python</i> and <i>Python</i> would have to be represented differently, as follows: 

``` 
\python and \Python
```

We can check these are different by running:

``` python
import re
re1 = re.compile('python')
print(bool(re1.match('Python')))
```

### 3.3 Disjunctions

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

### 3.4 Ranges

If we want a regular expression to express the disjunction of a range of characters, we can use a <b>dash</b>. For example, instead of the previous example, we can write 

``` 
[0-9]
```
Similarly, we can represent all characters of the alphabet with 

``` 
[a-z]
```

### 3.5 Exclusions

Brackets can also be used to represent what an expression <b>cannot</b> be if you combine it with the <b>caret</b> sign. For example, the expression 

``` 
[^p]
```
represents any character, special characters included, but p.

### 3.6 Question Marks 

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

### 3.7 Kleene Star

To represent the expressions containing zero or <b>more</b> instances of the previous character, we use an <b>asterisk</b> as the kleene star. To represent the set of strings containing <i>a, ab, abb, abbb, ...</i>, the following regular expression would be used:  
```
\ab*
```

### 3.8 Wildcards

Wildcards are used to represent the possibility of any character and symbolized with a <b>period</b>. For example, 

```
\beg.n
```
From this regular expression, the strings <i>begun, begin, began,</i> etc., can be generated. 

### 3.9 Kleene+

To represent the expressions containing at <b>least</b> one or more instances of the previous character, we use a <b>plus</b> sign. To represent the set of strings containing <i>ab, abb, abbb, ...</i>, the following regular expression would be used:  

```
\ab+
```

## 4.0 Word Tagging and Models

Given any sentence, you can classify each word as a noun, verb, conjunction, or any other class of words. When there are hundreds of thousands of sentences, even millions, this is obviously a large and tedious task. But it's not one that can't be solved computationally. 


### 4.1 NLTK Parts of Speech Tagger

NLTK is a package in python that provides libraries for different text processing techniques, such as classification, tokenization, stemming, parsing, but important to this example, tagging. 

``` python
import nltk 

text = nltk.word_tokenize("Python is an awesome language!")
nltk.pos_tag(text)
```

```python
[('Python', 'NNP'), ('is', 'VBZ'), ('an', 'DT'), ('awesome', 'JJ'), ('language', 'NN'), ('!', '.')]
```

Not sure what DT, JJ, or any other tag is? Just try this in your python shell: 

```python
nltk.help.upenn_tagset('JJ')
```
``` 
JJ: adjective or numeral, ordinal
third ill-mannered pre-war regrettable oiled calamitous first separable
ectoplasmic battery-powered participatory fourth still-to-be-named
multilingual multi-disciplinary ...
```


#### 4.1.1 Ambiguity

But what if a word can be tagged as more than one part of speech? For example, the word "sink." Depending on the content of the sentence, it could either be a noun or a verb.

Furthermore, what if a piece of text demonstrates a rhetorical device like sarcasm or irony? Clearly this can mislead the sentiment analyzer to misclassify a regular expression. 


### 4.2 Unigram Models

Remember our bag of words model from earlier? One of its characteristics was that it didn't take the ordering of the words into account - that's why we were able to use dictionaries to map each words to True values. 

With that said, unigram models are models where the order doesn't make a difference in our model. You might be wondering why we care about unigram models since they seem to be so simple, but don't let their simplicity fool you - they're a foundational block for a lot of more advanced techniques in NLP. 

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

### 4.3 Bigram Models

Here, ordering does matter. 

``` python
bigram_tagger = nltk.BigramTagger(brown_tagged_sents)
bigram_tagger.tag(brown_sents[2007])
```

Notice the changes from the last time we tagged the words of this same sentence: 

```
[('Various', 'JJ'), ('of', 'IN'), ('the', 'AT'), ('apartments', 'NNS'), ('are', 'BER'), ('of', 'IN'), ('the', 'AT'), ('terrace', 'NN'), ('type', 'NN'), (',', ','), ('being', 'BEG'), ('on', 'IN'), ('the', 'AT'), ('ground', 'NN'), ('floor', 'NN'), ('so', 'CS'), ('that', 'CS'), ('entrance', 'NN'), ('is', 'BEZ'), ('direct', 'JJ'), ('.', '.')]
```



## 5.0 Normalizing Text

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


### 5.1 Stemming

But we can do more! 

#### 5.1.1 What is Stemming?

Stemming is the process of converting the words of a sentence to its non-changing portions. In the example of amusing, amusement, and amused above, the stem would be amus.

#### 5.1.2 Types of Stemmers

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


### 5.2 Lemmatization

#### 5.2.1 What is Lemmatization?

Lemmatization is the process of converting the words of a sentence to its dictionary form. For example, given the words amusement, amusing, and amused, the lemma for each and all would be amuse.

#### 5.2.2 WordNetLemmatizer

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



Notice that women is changed to "woman"! 

## 6.0 Final Words 

Going back to our original sentiment analysis, we could have improved our model in a lot of ways by applying some of techniques we just went through. The twitter data is seemingly messy and inconsistent, so if we really wanted to get a highly accurate model, we could have done some preprocessing on the tweets to clean it up.

Secondly, the way in which we built our classifier could have been improved. Our feature extraction was relatively simple and could have been improved by using a bigram model rather than the bag of words model. We could have also fixed our Bayes Classifier so that it only took the most frequent words into considerations. 

### 6.1 Resources

[Natural Language Processing With Python](http://bit.ly/nlp-w-python) <br>
[Regular Expressions Cookbook](http://bit.ly/regular-expressions-cb)


Intermediate Natural Language Processing
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958). 

This guide assumes some basic knowledge of Natural Language Processing; more specifically, it assumes knowledge contained in [this](http://learn.adicu.com/nlp/) tutorial.

## Table of Contents

- [0.0 Setup](#00-setup)
    + [0.1 Python and Pip](#01-python--pip)
    + [0.2 Libraries](#02-libraries)
    + [0.3 Other](#03-other)
- [1.0 Background](#10-background)
    + [1.1 Polarity Flippers](#11-polarity-flippers)
    * [1.1.1 Negation](#111-negation)
    + [1.2 Multiword Expressions](#12-multiword-expressions)
    + [1.3 WordNet](#13-wordnet)
    * [1.3.1 Synsets](#131-synsets)
    * [1.3.2 Negation](#132-negations)
    + [1.4 SentiWordNet](#14-sentiwordnet)
    + [1.5 Stop Words](#15-stop-words)
    + [1.6 Testing](#16-testing)
    * [1.6.1 Cross Validation](#161-cross-validation)
    * [1.6.2 Precision](#162-precision)
    + [1.7 Logistic Regression](#17-logistic-regression)
- [2.0 Information Extraction](#20-information-extraction)
    + [2.1 Data Forms](#21-data-forms)
    + [2.2 What is Information Extraction?](#22-what-is-information-extraction)
- [3.0 Chunking](#30-chunking)
    + [3.1 Noun Phrase Chunking](#31-noun-phrase-chunking)
- [4.0 Named Entity Extraction](#40-named-entity-extraction)
    + [4.1 spaCy](#41-spacy)
    + [4.2 nltk](#42-nltk)
- [5.0 Relation Extraction](#50-relation-extraction)
    + [5.1 Rule-Based Systems](#51-rule--based-systems)
    + [5.2 Machine Learning](#52-machine-learning)
- [6.0 Sentiment Analysis](#60-sentiment-analysis)
    + [6.1 Loading the Data](#61-loading-the-data)
    + [6.2 Preparing the Data](#62-preparing-the-data)
    + [6.3 Linear Classifier](#63-linear-classifier)
- [7.0 Final Words](#70-final-words)
    + [7.1 Resources](#71-resources)
    + [7.2 Mini Courses](#72-mini-courses)

## 0.0 Setup

This guide was written in Python 3.6.

### 0.1 Python & Pip

If you haven't already, please download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).


### 0.2 Libraries

We'll be working with the re library for regular expressions and nltk for natural language processing techniques, so make sure to install them! To install these libraries, enter the following commands into your terminal: 

``` 
pip3 install nltk==3.2.4
pip3 install spacy==1.8.2
pip3 install pandas==0.20.1
pip3 install scikit-learn==0.18.1
```

### 0.3 Other

Sentence boundary detection requires the dependency parse, which requires data to be installed, so enter the following command in your terminal. 

```
python3 -m spacy.en.download all
```

### 0.4 Virtual Environment

If you'd like to work in a virtual environment, you can set it up as follows: 
```
pip3 install virtualenv
virtualenv your_env
```
And then launch it with: 
```
source your_env/bin/activate
```

To execute the visualizations in matplotlib, do the following:

```
cd ~/.matplotlib
vim matplotlibrc
```
And then, write `backend: TkAgg` in the file. Now you should be set up with your virtual environment!

Cool, now we're ready to start! 

## 1.0 Background

### 1.1 Polarity Flippers

Polarity flippers are words that change positive expressions into negative ones or vice versa. 

#### 1.1.1 Negation 

Negations directly change an expression's sentiment by preceding the word before it. An example would be

```
The cat is not nice.
```

#### 1.1.2 Constructive Discourse Connectives

Constructive Discourse Connectives are words which indirectly change an expression's meaning with words like "but". An example would be 

``` 
I usually like cats, but this cat is evil.
```

### 1.2 Multiword Expressions

Multiword expressions are important because, depending on the context, can be considered positive or negative. For example, 

``` 
This song is shit.
```
is definitely considered negative. Whereas

``` 
This song is the shit.
```
is actually considered positive, simply because of the addition of 'the' before the word 'shit'.

### 1.3 WordNet

WordNet is an English lexical database with emphasis on synonymy - sort of like a thesaurus. Specifically, nouns, verbs, adjectives and adjectives are grouped into synonym sets. 

#### 1.3.1 Synsets

nltk has a built-in WordNet that we can use to find synonyms. We import it as such:
``` python
from nltk.corpus import wordnet as wn
```

If we feed a word to the synsets() method, the return value will be the class to which belongs. For example, if we call the method on motorcycle,  

``` python
print(wn.synsets('motorcar'))
```

we get:

```
[Synset('car.n.01')]
```

Awesome stuff! But if we want to take it a step further, we can. We've previously learned what lemmas are - if you want to obtain the lemmas for a given synonym set, you can use the following method:

``` python
print(wn.synset('car.n.01').lemma_names())
```

This will get you:
```
['car', 'auto', 'automobile', 'machine', 'motorcar']
```

Even more, you can do things like get the definition of a word: 

``` python
print(wn.synset('car.n.01').definition())
```

Again, pretty neat stuff. 
```
'a motor vehicle with four wheels; usually propelled by an internal combustion engine'
```

#### 1.3.2 Negation

With WordNet, we can easily detect negations. This is great because it's not only fast, but it requires no training data and has a fairly good predictive accuracy. On the other hand, it's not able to handle context well or work with multiple word phrases. 


### 1.4 SentiWordNet

Based on WordNet synsets, SentiWordNet is a lexical resource for opinion mining, where each synset is assigned three sentiment scores: positivity, negativity, and objectivity.

``` python
from nltk.corpus import sentiwordnet as swn
cat = swn.senti_synset('cat.n.03')
```

``` python
cat.pos_score()
```

``` python
cat.neg_score()
```

``` python
cat.obj_score()
```

### 1.5 Stop Words

Stop words are extremely common words that would be of little value in our analysis are often excluded from the vocabulary entirely. Some common examples are determiners like the, a, an, another, but your list of stop words (or <b>stop list</b>) depends on the context of the problem you're working on. 

### 1.6 Testing

#### 1.6.1 Cross Validation

Cross validation is a model evaluation method that works by not using the entire data set when training the model, i.e. some of the data is removed before training begins. Once training is completed, the removed data is used to test the performance of the learned model on this data. This is important because it prevents your model from over learning (or overfitting) your data. 

#### 1.6.2 Precision

Precision is the percentage of retrieved instances that are relevant - it measures the exactness of a classifier. A higher precision means less false positives, while a lower precision means more false positives. 

#### 1.6.3 Recall

Recall is the percentage of relevant instances that are retrieved. Higher recall means less false negatives, while lower recall means more false negatives. Improving recall can often decrease precision because it gets increasingly harder to be precise as the sample space increases.

#### 1.6.4 F-measure 

The f1-score is a measure of a test's accuracy that considers both the precision and the recall. 

### 1.7 Logistic Regression

Logistic regression is a generalized linear model commonly used for classifying binary data. Its output is a continuous range of values between 0 and 1, usually representing the probability, and its input is some form of discrete predictor. 


## 2.0  Information Extraction

Information Extraction is the process of acquiring meaning from text in a computational manner. 

### 2.1 Data Forms

#### 2.1.1 Structured Data

Structured Data is when there is a regular and predictable organization of entities and relationships.

#### 2.1.2 Unstructured Data

Unstructured data, as the name suggests, assumes no organization. This is the case with most written textual data. 

### 2.2 What is Information Extraction?

With that said, information extraction is the means by which you acquire structured data from a given unstructured dataset. There are a number of ways in which this can be done, but generally, information extraction consists of searching for specific types of entities and relationships between those entities. 

An example is being given the following text, 

```
Martin received a 98% on his math exam, whereas Jacob received a 84%. Eli, who also took the same test, received an 89%. Lastly, Ojas received a 72%.
```
This is clearly unstructured. It requires reading for any logical relationships to be extracted. Through the use of information extraction techniques, however, we could output structured data such as the following: 

```
Name     Grade
Martin   98
Jacob    84
Eli      89
Ojas     72
```

## 3.0 Chunking

Chunking is used for entity recognition and segments and labels multitoken sequences. This typically involves segmenting multi-token sequences and labeling them with entity types, such as 'person', 'organization', or 'time'. 

### 3.1 Noun Phrase Chunking

Noun Phrase Chunking, or NP-Chunking, is where we search for chunks corresponding to individual noun phrases.

We can use nltk, as is the case most of the time, to create a chunk parser. We begin with importing nltk and defining a sentence with its parts-of-speeches tagged (which we covered in the previous tutorial). 

``` python
import nltk 
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"), ("dog", "NN"), ("barked", "VBD"), ("at", "IN"), ("the", "DT"), ("cat", "NN")]
```

Next, we define the tag pattern of an NP chunk. A tag pattern is a sequence of part-of-speech tags delimited using angle brackets, e.g. `<DT>?<JJ>*<NN>`. This is how the parse tree for a given sentence is acquired.  
``` python
pattern = "NP: {<DT>?<JJ>*<NN>}" 
```

Finally we create the chunk parser with the nltk RegexpParser() class. 
``` python
NPChunker = nltk.RegexpParser(pattern) 
```

And lastly, we actually parse the example sentence and display its parse tree. 
``` python
result = NPChunker.parse(sentence) 
result.draw()
```

## 4.0 Named Entity Extraction

Named entities are noun phrases that refer to specific types of individuals, such as organizations, people, dates, etc. Therefore, the purpose of a named entity recognition (NER) system is to identify all textual mentions of the named entities.

### 4.1 spaCy

In the following exercise, we'll build our own named entity recognition system with the Python module `spaCy`, a Python module commonly used for Natural Language Processing in industry. 

``` python
import spacy
import pandas as pd
```

Using spaCy, we'll load the built-in English tokenizer, tagger, parser, NER and word vectors. We indicate this with the parameter `'en'`:

``` python
nlp = spacy.load('en')
```

We need an example to actually process, so below is some text from Columbia's website:  

``` python
review = "Columbia University was founded in 1754 as King's College by royal charter of King George II of England. It is the oldest institution of higher learning in the state of New York and the fifth oldest in the United States. Controversy preceded the founding of the College, with various groups competing to determine its location and religious affiliation. Advocates of New York City met with success on the first point, while the Anglicans prevailed on the latter. However, all constituencies agreed to commit themselves to principles of religious liberty in establishing the policies of the College. In July 1754, Samuel Johnson held the first classes in a new schoolhouse adjoining Trinity Church, located on what is now lower Broadway in Manhattan. There were eight students in the class. At King's College, the future leaders of colonial society could receive an education designed to 'enlarge the Mind, improve the Understanding, polish the whole Man, and qualify them to support the brightest Characters in all the elevated stations in life.'' One early manifestation of the institution's lofty goals was the establishment in 1767 of the first American medical school to grant the M.D. degree."
```

With this example in mind, we feed it into the tokenizer.

``` python
doc = nlp(review)
```

Going along the process of named entity extraction, we begin by segmenting the text, i.e. splitting it into a list of sentences. 

``` python
sentences = [sentence.orth_ for sentence in doc.sents] # list of sentences
print("There were {} sentences found.".format(len(sentences)))
```

And we get: 

```
There were 9 sentences found.
```

Now, we go a step further, and count the number of nounphrases by taking advantage of chunk properties.

``` python
nounphrases = [[np.orth_, np.root.head.orth_] for np in doc.noun_chunks]
print("There were {} noun phrases found.".format(len(nounphrases)))
```

And we get:

```
There were 54 noun phrases found.
```

Lastly, we achieve our final goal: entity extraction. 

``` python
entities = list(doc.ents) # converts entities into a list
print("There were {} entities found".format(len(entities)))
```

And we get: 

```
There were 22 entities found
```

So now, we can turn this into a DataFrame for better visualization: 

``` python
orgs_and_people = [entity.orth_ for entity in entities if entity.label_ in ['ORG','PERSON']]
pd.DataFrame(orgs_and_people)
```

Unsurprisingly, Columbia University is an entity, along with other names like King's College and Samuel Johnson.

```
0  Columbia University      
1  King's College           
2  King George II of England
3  Samuel Johnson           
4  Trinity Church           
5  King's College 
```

In summary, named entity extraction typically follows the process of sentence segmentation, noun phrase chunking, and, finally, entity extraction. 

### 4.2 nltk

Next, we'll work through a similar example as before, this time using the nltk module to extract the named entities through the use of chunk parsing. As always, we begin by importing our needed modules and example:  

``` python
import nltk
import re
content = "Starbucks has not been doing well lately"
```

Then, as always, we tokenize the sentence and follow up with parts-of-speech tagging. 

``` python
tokenized = nltk.word_tokenize(content)
tagged = nltk.pos_tag(tokenized)
print(tagged)
```

Great, now we've got something to work with! 

``` 
[('Starbucks', 'NNP'), ('has', 'VBZ'), ('not', 'RB'), ('been', 'VBN'), ('doing', 'VBG'), ('well', 'RB'), ('lately', 'RB')]
```

So we take this POS tagged sentence and feed it to the `nltk.ne_chunk()` method. This method returns a nested Tree object, so we display the content with namedEnt.draw(). 

``` python
namedEnt = nltk.ne_chunk(tagged)
namedEnt.draw()
```

Now, if you wanted to simply get the named entities from the namedEnt object we created, how do you think you would go about doing so?

## 5.0 Relation Extraction 

Once we have identified named entities in a text, we then want to analyze for the relations that exist between them. This can be performed using either rule-based systems, which typically look for specific patterns in the text that connect entities and the intervening words, or using machine learning systems that typically attempt to learn such patterns automatically from a training corpus.

### 5.1 Rule-Based Systems

In the rule-based systems approach, we look for all triples of the form (X, a, Y), where X and Y are named entities and a is the string of words that indicates the relationship between X and Y. Using regular expressions, we can pull out those instances of a that express the relation that we are looking for. 

In the following code, we search for strings that contain the word "in". The special regular expression `(?!\b.+ing\b)` allows us to disregard strings such as `success in supervising the transition of`, where "in" is followed by a gerund. 

``` python
IN = re.compile(r'.*\bin\b(?!\b.+ing)')
for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
    for rel in nltk.sem.relextract.extract_rels('ORG', 'LOC', doc,corpus='ieer', pattern = IN):
        print (nltk.sem.relextract.rtuple(rel))
```

And so we get: 

```
[ORG: 'WHYY'] 'in' [LOC: 'Philadelphia']
[ORG: 'McGlashan &AMP; Sarrail'] 'firm in' [LOC: 'San Mateo']
[ORG: 'Freedom Forum'] 'in' [LOC: 'Arlington']
[ORG: 'Brookings Institution'] ', the research group in' [LOC: 'Washington']
[ORG: 'Idealab'] ', a self-described business incubator based in' [LOC: 'Los Angeles']
[ORG: 'Open Text'] ', based in' [LOC: 'Waterloo']
[ORG: 'WGBH'] 'in' [LOC: 'Boston']
[ORG: 'Bastille Opera'] 'in' [LOC: 'Paris']
[ORG: 'Omnicom'] 'in' [LOC: 'New York']
[ORG: 'DDB Needham'] 'in' [LOC: 'New York']
[ORG: 'Kaplan Thaler Group'] 'in' [LOC: 'New York']
[ORG: 'BBDO South'] 'in' [LOC: 'Atlanta']
[ORG: 'Georgia-Pacific'] 'in' [LOC: 'Atlanta']
```

Note that the X and Y named entitities types all match with one another! Object type matching is an important and required part of this process. 

### 5.2 Machine Learning

We won't be going through an example of a machine learning based entity extraction algorithm, but it's important to note the different machine learning algorithms that can be implemented to accomplish this task of relation extraction. 

Most simply, Logistic Regression can be used to classify the objects that relate to one another. But additionally, algorithms like Suport Vector Machines and Random Forest could also accomplish the job. Which algorithm you ultimately choose depends on which outperforms in terms of speed and accuracy.

In summary, it's important to note that while these algorithms will likely have high accurate rates, labeling thousands of relations (and entities!) is incredibly expensive.   


## 6.0 Sentiment Analysis

As we saw in the previous tutorial, sentiment analysis refers to the use of text analysis and statistical learning to identify and extract subjective information in textual data. For our last exercise in this tutorial, we'll introduce and use linear models in the context of a sentiment analysis problem.

### 6.1 Loading the Data

First, we begin by loading the data. Since we'll be using data available online, we'll use the `urllib` module to avoid having to manually download any data.

``` python
import urllib.request
```

Once imported, we'll then define the test and training data URLs as variables, as well as filenames for each of those datasets. This is so that we can easily download these to our local computer. 

``` python
test_url = "https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH-SI650/testdata.txt"
train_url = "https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH-SI650/training.txt"

test_file = 'test_data.csv'
train_file = 'train_data.csv'
```

Using the links and filenames from above, we'll officially download the data using the urlib.request.urlretrieve method. 

```
test_data_f = urllib.request.urlretrieve(test_url, test_file)
train_data_f = urllib.request.urlretrieve(train_url, train_file)
```

Now that we've downloaded our datasets, we can load them into pandas dataframes with the `read_csv` function. We'll start off with our test data and then repeat the same code for our training data. 

``` python
import pandas as pd

test_data_df = pd.read_csv(test_file, header=None, delimiter="\t", quoting=3)
test_data_df.columns = ["Text"]
```

The key difference here is that we set `.columns` to a list of two elements instead of one. This is because we need a column to indicate the label, otherwise the model won't be able to train. For our text data before, however, we explicitly don't want the training label since our model will be predicting those labels. 

``` python
train_data_df = pd.read_csv(train_file, header=None, delimiter="\t", quoting=3)
train_data_df.columns = ["Sentiment","Text"]
```

Just to see how the dataframe looks, let's call the `.head()` method on both dataframes. 

``` python
test_data_df.head()
```

And we get: 

```
Text
0  " I don't care what anyone says, I like Hillar...
1                  have an awesome time at purdue!..
2  Yep, I'm still in London, which is pretty awes...
3  Have to say, I hate Paris Hilton's behavior bu...
4                            i will love the lakers.
```


``` python
train_data_df.head()
```

And we get:

``` 
Sentiment                                               Text
0          1            The Da Vinci Code book is just awesome.
1          1  this was the first clive cussler i've ever rea...
2          1                   i liked the Da Vinci Code a lot.
3          1                   i liked the Da Vinci Code a lot.
4          1  I liked the Da Vinci Code but it ultimatly did...
```

### 6.2 Preparing the Data

To implement our bag-of-words linear classifier, we need our data in a format that allows us to feed it in to the classifer. Using sklearn.feature_extraction.text.CountVectorizer in the Python scikit learn module, we can convert the text documents to a matrix of token counts. So first, we import all the needed modules: 

``` python
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer        
from nltk.stem.porter import PorterStemmer
```

We need to remove punctuations, lowercase, remove stop words, and stem words. All these steps can be directly performed by CountVectorizer if we pass the right parameter values. We can do this as follows. 

We first create a stemmer, using the Porter Stemmer implementation.

``` python
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
stemmed = [stemmer.stem(item) for item in tokens]
return(stemmed)
```

Here, we have our tokenizer, which removes non-letters and stems:

``` python
def tokenize(text):
text = re.sub("[^a-zA-Z]", " ", text)
tokens = nltk.word_tokenize(text)
stems = stem_tokens(tokens, stemmer)
return(stems)
```

Here we init the vectoriser with the CountVectorizer class, making sure to pass our tokenizer and stemmers as parameters, remove stop words, and lowercase all characters.

``` python
vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = 'english',
    max_features = 85
)
```

Next, we use the `fit_transform()` method to transform our corpus data into feature vectors. Since the input needed is a list of strings, we concatenate all of our training and test data. 

``` python
features = vectorizer.fit_transform(
                          train_data_df.Text.tolist() + test_data_df.Text.tolist())
```

Here, we're simply converting the features to an array so we have an easier data structure to use.

``` python
features_nd = features.toarray()
```

### 6.3 Linear Classifier

Finally, we begin building our classifier. Earlier we learned what a bag-of-words model. Here, we'll be using a similar model, but with some modifications. To refresh your mind, this kind of model simplifies text to a multi-set of terms frequencies. 

So first we'll split our training data to get an evaluation set. As we mentioned before, we'll use cross validation to split the data. sklearn has a built-in method that will do this for us. All we need to do is provide the data and assign a training percentage (in this case, 85%).

``` python
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(
       features_nd[0:len(train_data_df)], 
       train_data_df.Sentiment,
       train_size=0.85, 
       random_state=1234)
```

Now we're ready to train our classifier. We'll be using Logistic Regression to model this data. Once again, sklearn has a built-in model for you to use, so we begin by importing the needed modules and calling the class.  

``` python
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
```

And as always, we need actually do the training, so we call the `.fit()` method on our data. 

``` python
log_model = log_model.fit(X=X_train, y=y_train)
```

Now we use the classifier to label the evaluation set we created earlier:

``` python
y_pred = log_model.predict(X_test)
```

You can see that this array of labels looks like: 

```
array([0, 1, 0, ..., 0, 1, 0])
```

### 6.4 Accuracy

In sklearn, there is a function called sklearn.metrics.classification_report which calculates several types of predictive scores on a classification model. So here we check out how exactly our model is performing:

``` python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

And we get: 

```
precision    recall  f1-score   support

0       0.98      0.99      0.98       467
1       0.99      0.98      0.99       596

avg / total       0.98      0.98      0.98      1063
```

where precision, recall, and f1-score are the accuracy values discussed in the section 1.6. Support is the number of occurrences of each class in y_true and x_true.


### 6.5 Retraining 

Finally, we can re-train our model with all the training data and use it for sentiment classification with the original unlabeled test set. 

So we repeat the process from earlier, this time with different data:

``` python
log_model = LogisticRegression()
log_model = log_model.fit(X=features_nd[0:len(train_data_df)], y=train_data_df.Sentiment)
test_pred = log_model.predict(features_nd[len(train_data_df):])
```

So again, we can see what the predictions look: 

```
array([1, 1, 1, ..., 1, 1, 0])
```

And lastly, let's actually look at our predictions! Using the random module to select a random sliver of the data we predicted on, we'll print the results.  

``` python
import random
spl = random.sample(range(len(test_pred)), 10)
for text, sentiment in zip(test_data_df.Text[spl], test_pred[spl]):
    print(sentiment, text)
```

Recall that 0 indicates a negative sentence and 1 indicates a positive:

```
0 harvard is dumb, i mean they really have to be stupid to have not wanted her to be at their school.
0 I've been working on an article, and Antid Oto has been, er, so upset about the shitty Harvard plagiarizer that he hasn't been able to even look at keyboards.
0 I hate the Lakers...
0 Boston SUCKS.
0 stupid kids and their need for Honda emblems):
1 London-Museums I really love the museums in London because there are a lot for me to see and they are free!
0 Stupid UCLA.
1 as title, tho i hate london, i did love alittle bit about london..
1 I love the lakers even tho Trav makes fun of me.
1 that i love you aaa lllooootttttt...
```


## 7.0 Final Words 

Remembering the sentiment analysis we performed with the Naive Bayes Classifier, we can see that the Logistic Regression classifier performs better with accuracy rates of 98%. 

You might be asking yourself why this is - remember that the Naive Bayes Classifier was a unigram model, in that it failed to consider the words that preceded. 


### 7.1 Resources

[Natural Language Processing With Python](http://bit.ly/nlp-w-python) <br>
[Regular Expressions Cookbook](http://bit.ly/regular-expressions-cb)



