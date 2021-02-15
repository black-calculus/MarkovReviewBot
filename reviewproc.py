# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:48:42 2020

@author: Damien Smith
"""

import language_tool_python

grab_next_line = False

reviews_collection = []

with open('reviewproc.txt', 'r') as f:
    for line in f:
        if grab_next_line:
            reviews_collection.append(line)
            grab_next_line = False
        if  " >>> " in line:
            grab_next_line = True
            
print(len(reviews_collection))

lang_tool = language_tool_python.LanguageTool('en-US')
grammar_err_dataset = 0

print("Calculating grammar errors in generated reviews...")

for i in range(len(reviews_collection)):
    gramm_err = lang_tool.check(reviews_collection[i])
    grammar_err_dataset += len(gramm_err)

print("Calculated.")
print("Average grammar errors/review from all", len(reviews_collection), "reviews: " + str(grammar_err_dataset/len(reviews_collection)))