# falsify

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/whVWeVGo)
# Assignment 3: Vector Space Query
**Westmont College Fall 2023**

**CS 128 Information Retrieval and Big Data**

*Assistant Professor* Mike Ryu (mryu@westmont.edu) 

## Autor Information
* **Name(s)**: Garrett Buchanan
* **Email(s)**: gbuchanan@westmont.edu

## Problem Description

The problem for this assignment was being able to implement three classes; Vector, Document, and Corpus, and write methods
within them that allowed for manipulation, as well as calculations, with a final goal of being able to create scores for a
corpus so that it can be ranked properly when given a query. In the context of this assignment this code should be able to
run and populate a pickle file that contains the speeches of all the presidents, then return ranked results according to a
user's query. In addition to writing code that can do all this, writing python unit tests that ensure the methods are
implemented correctly is also necessary.

## Description of the Solution

To solve this problem I implemented methods for the Vector, Document, and Corpus classes. For Vector class I implemented methods
to compute the euclidian norm of a given vector, to compute the dot product of two vectors, and to determine the cossimilarity
score of two given vectors. For the Document class I implemented methods that filter a given stop word out of a document, that stems
the words in a given document in order to not skew ranking by only using the root word, and term frequency which returns how many
times a given term occurs within a document. For the Corpus Class I implemented methods that compute a unique corpus as an indexed
dictionary, checking if a term occurs in a list of words withing a doc, computing the document frequencies of every term in a document,
computing a tf-idf vector for a document, and computing the tf-idf matrix for an entire corpus

## Key Takeaways

I learned that I can actually thoroughly enjoy the coding process. Lately coding has felt like a huge stressor, but this assignment
was great because although it was still challenging I was able to figure things out by taking time to play around with different
possible solutions. In the context of my coding knowledge gained on this assignment, I feel much more comfortable implementing
methods and referencing them within other functions. I also learned a ton about how to translate mathematical functions and equations
into code as well as using trigonomic functions in the context of code. I also gained a lot of knowledge whist practicing implementation
of while, and for loops, along with increased proficiency in if statements. I also feel like this is the first computer science assignment
I have completed completely on my own that I can be super proud of. This assignment is really cool and I'm going to be showing it to
everyone that I know haha.
