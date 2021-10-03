# FastText Stackoverflow - version 5.0

# import streamlit libraries
import streamlit as st

# import os library
import os

# Upgrade pip
os.system('/home/appuser/venv/bin/python -m pip install --upgrade pip')
st.write('>> pip has been upgraded...')

# scipy installation
os.system('pip install scipy')
st.write('>> scipy has been upgraded...')

# ! pwd
import subprocess
p = subprocess.Popen(["pwd"], stdout=subprocess.PIPE)
output, err = p.communicate()
st.write("*** Running pwd command ***\n", output.decode('ascii'), err)

# Installing FastText

# Cloning fastText from GitHub
os.system('git clone https://github.com/facebookresearch/fastText.git')
st.write("Cloned")

p = subprocess.Popen(["pwd"], stdout=subprocess.PIPE)
output, err = p.communicate()
st.write("*** Running pwd command ***\n", output.decode('ascii'), err)

p = subprocess.Popen(["ls", "-l"], stdout=subprocess.PIPE)
output, err = p.communicate()
st.write("*** Running ls -l command ***\n", output.decode('ascii'), err)
