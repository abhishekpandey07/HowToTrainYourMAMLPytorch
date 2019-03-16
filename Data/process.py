import re
import os
import glob
import sys

"""
Preprocess files for PAN-2018 dataset

Usage: 
    $ python process.py dataDir
"""

def clean_stories(str):

    txt = str.replace("\n"," ") # remove new line 
    txt = re.sub(r'''[^a-zA-Z0-9,.!?’'\"]''', ' ', txt) # remove special chars
    txt = " ".join(re.split("\s+", txt)) # remove duplicate white spaces
    txt = re.sub(r'\d+', '@', txt) # convert all numbers to @
    return txt

def char_500(txt):

    n = 500    
    txt = [txt[i:i+n] for i in range(0,len(txt),n)]

    return txt

def process(txt):
    txt = clean_stories(txt)

    txt = char_500(txt)
    return txt

def make_file(filename):
    with open(filename,"r") as input, open("{}.processed".format(filename),"w") as output:
        txt = input.read()
        txt_list = process(txt)

        done = "\n".join(txt_list)
   
        output.write(done)
        

if __name__ == "__main__":
    dataDir = sys.argv[1]
    
    problems = glob.glob("{}/*/*/known*".format(dataDir))
    for p in problems:
        make_file(p)
    
    problems = glob.glob("{}/*/*/*".format(dataDir))
    print(problems)
    for p in problems:
        make_file(p)
        

 

