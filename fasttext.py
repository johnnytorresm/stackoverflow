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

# os.chdir("fastText")
# st.write("Working directory changed to fastText...")

# os.system('ls -l')

from pathlib import Path

path = Path(os.getcwd())
parent = str(path.parent.absolute())

original_file = parent + "/corpus10k.csv"
cmd = "cp " + original_file + " ./fasText "
os.system(cmd)

original_file = parent + "/other-stop-words.txt"
cmd = "cp " + original_file + " ./fastText "
os.system(cmd)

st.write("*** copying files ***\n", output.decode('ascii'), err)

os.system('ls -l *.txt')
os.system('ls -l *.csv')

