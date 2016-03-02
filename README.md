Intro to Natural Language Processing
==============================================

http://learn.adicu.com/nlp

Learn the basics of Natural Language Processing!

### Data

The data files `pos_tweets.txt` and `neg_tweets.txt` contain text data used in this tutorial.

### Building

Run the following in the root directory:

    make

This generates an `output.html` file to be viewed in a browser.

### Deploying

Run the following command to deploy to [learn.adicu.com/nlp](http://learn.adicu.com/nlp) (requires SCP and access to adi-website on SSH):

    make deploy

### Directory Structure

#### build/

This is where all the extra files needed to convert from markdown to HTML go. `Makefile` uses the files from this folder.

