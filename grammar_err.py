# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:45:57 2020

@author: Damien Smith
"""
import language_tool_python

import random

import pandas as pd

# ================== DATASET ================== #
data=pd.read_csv('IMDB_Dataset_Quantized.csv')
# ============================================= #

# ================== OTHER STUFF ================= #
lang_tool = language_tool_python.LanguageTool('en-US')
# ================================================ #

random.seed(100)

grammar_err_dataset = 0

print("Calculating grammar errors in dataset...")

for i in range(600):
    j = random.randint(0,49999)
    gramm_err = lang_tool.check(data['review'][j])
    grammar_err_dataset += len(gramm_err)

print("Calculated.")
print("Average grammar errors/review from 600 randomly-sampled reviews: " + str(grammar_err_dataset/600))
