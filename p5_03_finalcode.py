#----------------------------------------------------
#   p5_03_finalcode.py
#----------------------------------------------------
# Open Classrooms
# Machine Learning Engineering Cursus  
# Student: Johnny TORRES
# Mentor: Julien HEDUIK
#----------------------------------------------------
# Projet 5: Catégorisez automatiquement des questions
#----------------------------------------------------

# FastText Stackoverflow - version 5.0

#-----------------------------
# Importing libraries
#-----------------------------
import streamlit as st
import os
import numpy as np 
import pandas as pd
import re
import datetime
import glob
import string 
import io
import csv
import codecs

import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

from datetime import timedelta

from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import MultiLabelBinarizer

from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score

import ast
from collections import Counter

# Upgrade pip
os.system('/home/appuser/venv/bin/python -m pip install --upgrade pip')

# Cloning fastText from Facebook Research GitHub
os.system('git clone https://github.com/facebookresearch/fastText.git')
os.chdir("fastText")

# Building the fasttext modules
os.system('make')

original_file = "/app/stackoverflow/other-stop-words.txt"
cmd = "cp " + original_file + " . "
os.system(cmd)

original_file = "/app/stackoverflow/corpus25k.csv"
cmd = "cp " + original_file + " . "
os.system(cmd)

#--------------------
# Global Variables
#--------------------
sw  = set(stopwords.words())

# Exhaustive list of English prepositions to be added to stop words
preps = pd.read_csv('other-stop-words.txt', header=None, names=['other-stop-words'])
preps_set = set(preps['other-stop-words'].tolist())
sw |= preps_set

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

#-----------------------------------------
# Function definitions
#-----------------------------------------

#----------------------------------------------------
# clean_txt: Cleans a string, and then lemmatize it.
#----------------------------------------------------
def clean_txt(txt): 
    # txt: string to clean
    
    global sw, lemmatizer                # sw = English stop words; lemmatizer from WordNet

    # 1. Lowercase 
    txt = txt.lower()

    # 2. Strip HTML code
    txt = re.sub('<[^<]+?>', ' ', txt)

    # 3. Strip punctuation signs
    txt = re.sub('[^\w\s]', ' ', txt)
    txt = re.sub('\*', ' ', txt)

    # 4. remove stop words
    txt_set = set(nltk.word_tokenize(txt))
    txt_lst = [w for w in txt_set if (not w in sw)]
    txt_set = set(txt_lst)
    txt_lst = list(txt_set)
    
    # 5. Lemmatisation
    txt_lst = [lemmatizer.lemmatize(w, pos='v') for w in txt_lst]
    txt = ' '.join([lemmatizer.lemmatize(w, pos='n') for w in txt_lst])
    
    return txt
    
#-----------------------------------------
# Receives a list of tags and returns
# a list of Labels in fasttext format
#-----------------------------------------
def labels(Tags):
    y_labels = ' '
    for t in Tags:
        y_labels += '__label__'+ t + ' '
    return y_labels

#-----------------------------------------
# most frequent tags in the corpus
#-----------------------------------------
def most_used_tags(tags, top_tags):
    final_tags = []
    for tag in tags:
        if tag in top_tags['Tags'].values:
            final_tags.append(tag)
    return final_tags
    
#-----------------------------------------
# Remove "__label__" from FastText Labels
#-----------------------------------------
def remove_label(l,prefix)-> str:
    if l.startswith(prefix):
        return l[len(prefix):]
    else:
        return l[:]

#---------------------------------------------
# Sorts a dictionary by value. Returns a list.
#---------------------------------------------
def sort_tags(x):
    if len(x)>0:
        s = sorted(x.items(), key = lambda y:(y[1], y[0]), reverse = True)
    else:
        s = []
    return s

#---------------------------------------------
# Top 5 values 
#---------------------------------------------
def top_tags(x):
    s = []
    i = 0
    if len(x)>0:
        while ((i<len(x)) and (i<5)):
            s = s + [x[i][0]]
            i += 1
    else:
        s = []
    return s

#--------------------------------
# Reading data
#--------------------------------
file2open = "corpus25k.csv"
posts = pd.read_csv(file2open, usecols=['Id', 'Tags', 'Text'])

# # Number of records read
# st.write(posts.shape)
# # st.write(posts.head())
        
# # counting nulls per column
# st.write(posts.isnull().sum())

#----------------------------------------------------------------
# Checking if Tags have been read as a single string or a series
#-----------------------------------------------------------------
if isinstance(posts['Tags'].iloc[0],str): # verifies if 'Tags' is of type "string"
    posts['Tags'] = posts['Tags'].apply(lambda tag: ast.literal_eval(tag))

#------------------------------------------------------
# Flatten the "Tags" columns into a Series and then 
# count the number of occurrences per item, putting
# the results into a dataframe
#------------------------------------------------------
tag_series = pd.Series([item for sublist in posts['Tags'] for item in sublist])
tag_df = tag_series.groupby(tag_series).size().rename_axis('Tags').reset_index(name='Nº of occurrences')
tag_df = tag_df.sort_values(by=['Nº of occurrences'], ascending=False)

# Nº of Tags that appear more than 100 times across all messages
series = tag_df['Nº of occurrences'].apply(lambda x: True if x > 10 else False)
counting = len(series[series == True].index)

#-----------------------------------------
# Top 100 tags
#-----------------------------------------
top_tags = tag_df[['Tags', 'Nº of occurrences']].head(100)

#--------------------------
# y = list of Tags list
#--------------------------
posts['Tags'] = posts['Tags'].apply(lambda tags: most_used_tags(tags, top_tags))
posts = posts.loc[posts['Tags'].str.len() > 0]
y = posts['Tags']

#------------------------------------
# Apply MultiLabelBinarizer to Tags
#------------------------------------
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(posts['Tags'])

#------------
# TD-IDF 
#------------
tfidf = TfidfVectorizer(analyzer="word", max_features=1000, ngram_range=(1,1))
X = tfidf.fit_transform(posts['Text'])

#------------------------------
# Split into train and test
#------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape,  X_test.shape,  y_train.shape, y_test.shape
st.write('Splitted into train and test')

#---------------------------------------------------------------------------------------------------------
# Preparation of tags for fasttext 
# The string "__label__" is attached to each tag to create a new column named 'Labels'.
# Columns 'Labels' and 'Text' are then written to an external txt file for fasttext analysis (in OS mode)
#----------------------------------------------------------------------------------------------------------
train, test = train_test_split(posts, test_size=0.2)
train['Labels'] = train['Tags'].apply(lambda x: labels(x))
test['Labels'] = test['Tags'].apply(lambda x: labels(x))
train = train.dropna()
test = test.dropna()

#-------------------------------------------------------
# It writes Labels and Text in FastText format to train.txt and test.txt files
# Those files will be the input for FastText routines
#-------------------------------------------------------
train_file = os.path.splitext(file2open)[0] + '.train'
test_file  = os.path.splitext(file2open)[0] + '.test'

train.to_csv(train_file, 
             header=None, 
             index=None, 
             mode='a', 
             encoding='utf-8', 
             columns=['Labels', 'Text'],
             sep=' ',
             escapechar=' ',
             quoting=csv.QUOTE_NONE,
             quotechar=' '
            )

test.to_csv(test_file,
            header=None, 
            index=None, 
            mode='a', 
            encoding='utf-8', 
            columns=['Labels', 'Text'],
            sep=' ',
            escapechar=' ',
            quoting=csv.QUOTE_NONE,
            quotechar=' '
           )

#-------------------------------------------------------
# Training the FastText model
#-------------------------------------------------------
# Command for training 
import subprocess
filename = os.path.splitext(file2open)[0]
input_file = filename + '.train'
output_file = filename
result_file = filename + ".result"
batcmd = './fasttext supervised -input '+ input_file + ' -output ' + \
                                          output_file + ' -dim 10 -lr 1 -wordNgrams 1 -minCount 1 -bucket 10000000 -epoch 25' + \
                                          '> ' + result_file
result = subprocess.check_output(batcmd, shell=True)

#-------------------------------------------------------
# Predicting labels
#-------------------------------------------------------
# File where question is written
question_file = open("question.txt", "w")

# File that contains the model
input_file = filename + '.bin'

# User is asked
question = st.text_area("Please, enter your post below", "")

if len(question)>0:
    st.write('The question was:')
    st.write(question)
# Question is cleaned and lemmatized
    question2 = clean_txt(question)
# Question is written to a file
    n = question_file.write(question2)
# Question file is closed
    question_file.close()
# FastText string command is assembled
    batcmd = './fasttext predict '+ input_file + ' question.txt 5 1>labels.csv'
# FastText commnd is executed. Output written to "labels.csv"
    result = subprocess.check_output(batcmd, shell=True)
# labels are read and split into a list
    csv_reader = csv.reader(codecs.open('labels.csv', 'rU', 'utf-8'))
    lists_from_csv = []
    for row in csv_reader:
        lists_from_csv.append(row)
    list_from_csv = list(tuple(remove_label(l,'__label__') for l in lists_from_csv[0][0].split()))
    st.write(list_from_csv)
