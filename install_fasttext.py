# Installing FastText - version 1.0

# import streamlit
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

os.system('cp corpus25k.csv ./fastText')
os.system('cp other-stop-words.txt ./fastText')

os.system('cd ./fastText')
os.system('pwd')
os.system('ls -l')
