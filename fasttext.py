# FastText Stackoverflow - version 5.0# import streamlit
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
st.write("Directory changed...")

p = subprocess.Popen(["ls", "-l"], stdout=subprocess.PIPE)
output, err = p.communicate()
print("*** Running ls -l command ***\n", output.decode('ascii'), err)

from pathlib import Path

path = Path(os.getcwd())
parent = str(path.parent.absolute())

original_file = parent + "/corpus10k.csv"
p = subprocess.Popen(["cp", original_file, "."], stdout=subprocess.PIPE)
output, err = p.communicate()
st.write("*** copying files ***\n", output.decode('ascii'), err)

original_file = parent + "/other-stop-words.txt"
p = subprocess.Popen(["cp", original_file, "."], stdout=subprocess.PIPE)
output, err = p.communicate()
st.write("*** copying files ***\n", output.decode('ascii'), err)

p = subprocess.Popen(["ls", "-l"], stdout=subprocess.PIPE)
output, err = p.communicate()
st.write("*** Running ls -l command ***\n", output.decode('ascii'), err)
