#----------------------------------------------------
#   p5_03_finalcode.py
#----------------------------------------------------
# Open Classrooms
# Machine Learning Engineering Cursus  
# Student: Johnny TORRES
# Mentor: Julien Heduik
#----------------------------------------------------
# Projet 5: Catégorisez automatiquement des questions
#----------------------------------------------------

#----------------------------------------------------
#   Importing libraries
#----------------------------------------------------
import streamlit as st
from pathlib import Path

import numpy as np 
import pandas as pd
import re
import datetime
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import csv
import codecs

import os
import io
import subprocess
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from datetime import timedelta

from ipywidgets import *
from tkinter import Tk, filedialog
from IPython.display import clear_output, display

import warnings
warnings.filterwarnings("ignore")

st.write("## Open Classrooms - Machine Learning Engineering")
st.write("# Project 5 - Tag assignment for Stack Overflow")
st.write("### Student: Johnny Torres  / Mentor: Julien Heduik\n")

st.write('Libraries have been imported')

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

st.write('Global Variables have been defined')

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
    txt = ' '.join([lemmatizer.lemmatize(w, pos='v') for w in txt_lst])
    
    return txt
    
#-----------------------------------------
# Remove "__label__" from FastText Labels
#-----------------------------------------
def remove_label(l,prefix)-> str:
    if l.startswith(prefix):
        return l[len(prefix):]
    else:
        return l[:]

#---------------------------------------------
# Tags question entered by user
#---------------------------------------------

def tagging():
    user_input = st.text_area("Enter your question here below", "end", key = str(counter))
    question_txt = clean_txt(user_input)
    label_list = []
    if (question_txt == "end"):
        return label_list, False
    else:
    # Question is written to a file
        question_file = open(r"question.txt","w")
        n = question_file.write(question_txt)
        # Question file is closed
        question_file.close()

        # FastText string command is assembled. Output written to "lbl_lst.csv"
        batcmd = 'fasttext predict train-corpus25k.bin' + ' question.txt 5 1>lbl_lst.csv'

        # FastText commnd is executed
        result = subprocess.check_output(batcmd, shell=True)

        # labels are read and split into a list
        labels = csv.reader(codecs.open('lbl_lst.csv', 'rU', 'utf-8'))
        list_from_csv =list(labels)
        if len(list_from_csv)>0:
            label_list = list_from_csv[0]
            label_list = label_list[0].split()
            for i in range(0,len(label_list)):
                label_list[i] = remove_label(label_list[i], '__label__')
        return label_list, True
        
#--------------------
# Main program
#--------------------

# Initialization of variables

counter=0
keep_going = True

# Main Loop
while (keep_going):
    results, keep_going = tagging()
    if (len(results)!=0):
        for l in results:   
            st.write(l, end=' ')
    counter += 1
    if not(keep_going):
        st.write("End of Project")
        st.stop()
st.stop()
