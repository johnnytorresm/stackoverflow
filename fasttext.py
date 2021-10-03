# FastText Stackoverflow - version 5.0
import streamlit as st

# import os library
import os

# Upgrade pip
os.system('/home/appuser/venv/bin/python -m pip install --upgrade pip')
st.write('pip has been upgraded...')

# Cloning fastText from Facebook Research GitHub
os.system('git clone https://github.com/facebookresearch/fastText.git')

# Building the fasttext modules
os.system('make')

# Verification
os.system('pwd')
os.system('ls -l')

# scipy installation
os.system('pip install scipy')

st.write(' FastText has been installed...')

os.chdir("fastText")
st.write("Working directory changed to fastText...")

# os.system('ls -l')

original_file = "https://github.com/johnnytorresm/stackoverflow/blob/main/other-stop-words.txt"
cmd = "cp " + original_file + " . "
os.system(cmd)

original_file = "https://github.com/johnnytorresm/stackoverflow/blob/main/corpus25k.csv"
cmd = "cp " + original_file + " . "
os.system(cmd)

st.write("*** copying files ***\n")

os.system('ls -l *.txt')
os.system('ls -l *.csv')

st.write('The End')
