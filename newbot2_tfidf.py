# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:03:25 2020

@author: Max Cooper / Anita Mallik / Damien Smith
"""
# ===================================================================================================================== #
# ===================================================================================================================== #
# ===================================================================================================================== #
# ===================================================================================================================== #
# ===================================================================================================================== #

# ========================================== IMPORTS ========================================== #

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split  # Split the training and testing data

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

import markovify  # To generate random sentences/reviews

import numpy as np

import language_tool_python

import sys
# ============================================================================================= #

# ===================================================================================================================== #
# ===================================================================================================================== #
# ===================================================================================================================== #
# ===================================================================================================================== #
# ===================================================================================================================== #



# ============================== CONSTANTS ============================== #
review_num = 8    # Number of review fragments to generate for each test
sentence_num = 1  # Number of sentences to generate per review fragment
# ======================================================================= #

# ================== DATASET ================== #
data=pd.read_csv('IMDB_Dataset_Quantized.csv')
# ============================================= #

# ================== OTHER STUFF ================= #
lang_tool = language_tool_python.LanguageTool('en-US')
# ================================================ #

"""
grammar_err_dataset = 0

print("Calculating grammar errors in dataset...")

for i in range(100):
    gramm_err = lang_tool.check(data['review'][i])
    grammar_err_dataset += len(gramm_err)

print("Calculated.")
print("Average grammar errors/review from 100 randomly-sampled reviews: " + str(grammar_err_dataset/100))
"""

"""
Sentiment_count=data.groupby('sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['review'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()
"""




# === Utilizing TF-IDF === #
# Make a TF-IDF vectorizer to convert our the text from the reviews in the data set to a sparse matrix of features
tf=TfidfVectorizer()
text_counts= tf.fit_transform(data['review'])

X_train, X_test, y_train, y_test = train_test_split(
    text_counts, data['sentiment'], test_size=0.3, random_state=154)


# === Markov Chain trained on entire data set === #
text_model = markovify.Text(data.review, state_size = 3)


# === Markov Chain trained on positive/negative separated data sets === #
# Separate our data
positive_review_dataset = []
positive_review_dataset_sentiments = []
negative_review_dataset = []
negative_review_dataset_sentiments = []

for i in range(len(data.sentiment)):
    if data.sentiment[i] == 1:
        positive_review_dataset.append(data.review[i])
        positive_review_dataset_sentiments.append(data.sentiment[i])
    else:
        negative_review_dataset.append(data.review[i])
        negative_review_dataset_sentiments.append(data.sentiment[i])


# Generate our small review segments

# Positive reviews Markov Chain
positive_text_model = markovify.Text(positive_review_dataset, state_size = 3)

# Negative reviews Markov Chain
negative_text_model = markovify.Text(negative_review_dataset, state_size = 3)


# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)


"""
# Model Generation Using Multinomial Naive Bayes
clf_mul = MultinomialNB().fit(X_train, y_train)
predicted= clf_mul.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))

# Model Generation Using Gaussian Naive Bayes
# Requires dense matrix, sparse matrix generated by Vectorizer insufficient
clf_gauss = GaussianNB().fit(X_train, y_train)
predicted= clf_gauss.predict(X_test)
print("GaussianNB Accuracy:",metrics.accuracy_score(y_test, predicted))

# Model Generation Using Complement Naive Bayes
clf_comp = ComplementNB().fit(X_train, y_train)
predicted= clf_comp.predict(X_test)
print("ComplementNB Accuracy:",metrics.accuracy_score(y_test, predicted))

# Model Generation Using Bernoulli Naive Bayes
clf_bern = BernoulliNB().fit(X_train, y_train)
predicted= clf_bern.predict(X_test)
print("BernoulliNB Accuracy:",metrics.accuracy_score(y_test, predicted))
"""



original_stdout = sys.stdout # Save a reference to the original standard output

# List to hold our number of positive classifications from the positively-trained Markov chain for average positive classification accuracy
pos_class_metric = []

# Variables to hold the number of reviews big generated based on positively-classified review fragments, and the number of those big reviews that the Naive Bayes model classified as positive
# Whole dataset
pos_conf_metric = 0
pos_conf_num = 0
# Split dataset
pos_conf_metric_split = 0
pos_conf_num_split = 0

# List to hold our number of negative classifications from the negatively-trained Markov chain for average negative classification accuracy
neg_class_metric = []

# Variables to hold the number of big reviews generated based on negatively-classified review fragments, and the number of those big reviews that the Naive Bayes model classified as negative
# Whole dataset
neg_conf_metric = 0
neg_conf_num = 0
# Split dataset
neg_conf_metric_split = 0
neg_conf_num_split = 0

# List to hold all of our reviews for grammar error checking
review_list_grammar_check = []
grammar_err_generated = 0

# We're ready to start, write to the console that we're doing work...
print(" ===> Working...")

# Write all output to 'markov_runs.txt'
with open('markov_runs.txt', 'w', errors='ignore') as f:
    sys.stdout = f # Change the standard output to the file we created
    
    # Output the accuracy of the Multinomial NB model on the data set
    print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
    print("\n\n")
    
    
    
    
    for i in range(100):
        # Label the current run
        print(" = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =")
        print(" - - - - - - - - - - - - - - - - - - [ ==================== RUN", i+1, "==================== ] - - - - - - - - - - - - - - - - - -")
        print(" = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =")
        
        print("Starting...")
        
        # ================================================================= #
        # ================================================================= #
        # ========== Markov Chain trained on the entire data set ========== #        
        # ================================================================= #
        # ================================================================= #
        print(" ===================================================== ")
        print(" ========== ALL-REVIEW-TRAINED MARKOV CHAIN ========== ")
        print(" ===================================================== ")
        
        # Make a list to hold each sentence generated
        sentences_list = []
        for i in range(review_num):
            # Initialize the new review fragment
            new_review = ""
            for j in range(sentence_num):
                # Append the new sentence to the review fragment
                new_review += text_model.make_sentence() + " "
            
            sentences_list.append(new_review)
        
        # Make a TF-IDF vectorizer to convert our sentence into a list of features
        ns_cv = TfidfVectorizer()
        # Generate the feature list matrix based on the list of review fragments
        feature_list = ns_cv.fit_transform(sentences_list)
        feature_list.resize(feature_list.shape[0],text_counts.shape[1])  # Resize the sparse matrix 'feature list' of the current review fragment to fit into the trained model
        
        # Classify each review fragment
        classification = clf.predict(feature_list)
            
        
        # Empty list to hold the positively-classified review fragments
        positive_reviews = []
        # Empty list to hold the negative reviews
        negative_reviews = []
    
        # Print the classification of each review fragment        
        for i in range(review_num):
            print("Fragment " + str(i+1) + ":\n" + sentences_list[i])
            if classification[i] == 1:
                print("Classification: Positive")
                positive_reviews.append(sentences_list[i])
            else:
                print("Classification: Negative")
                negative_reviews.append(sentences_list[i])
            print("")
        
        # Print our big positive review based on all of the positively-classified fragments
        positive_movie_review = "" # Initialize the review
        for small_review in positive_reviews:
            positive_movie_review += small_review # Append each positively-classified fragment to the positive review
        
        print(" >>> Long Review 1 (positive-classified fragments):")
        print(positive_movie_review)
        pmv_list = [positive_movie_review] # Place the review into a list so that it can be vectorized by the TF-IDF vectorizer for classification
        review_list_grammar_check.append(positive_movie_review) # Append to the list of all reviews to check for grammar errors later
        print("")
        
        # Print our big negative review based on all of the positively-classified fragments
        negative_movie_review = ""
        for small_review in negative_reviews:
            negative_movie_review += small_review # Append each negatively-classified fragment to the positive review
        
        print(" >>> Long Review 2 (negatively-classified fragments):")
        print(negative_movie_review)
        nmv_list = [negative_movie_review] # Place the review into a list so that it can be vectorized by the TF-IDF vectorizer for classification
        review_list_grammar_check.append(negative_movie_review) # Append to the list of all reviews to check for grammar errors later
        print("")
        
        
        # Classify the large positive review
        # Skip if empty
        if pmv_list[0] != '':
            ns_cv_p = TfidfVectorizer() # TF-IDF vectorizer for the large positive review
            feature_list = ns_cv_p.fit_transform(pmv_list)
            
            feature_list.resize(feature_list.shape[0],text_counts.shape[1])  # Resize the sparse matrix 'feature list' of the current sentnce to fit into the trained model
            
            classification = clf.predict(feature_list)
            pos_conf_num +=1 # Add one to the list of positive reviews generated
            if classification[0] == 1:
                print(" ~~~ Review 1 Classification: Positive")
                pos_conf_metric += 1 # Correctly confirmed as a positive review; add one to the number of correct confirmations for positive reviews
            else:
                print(" ~~~ Review 1 Classification: Negative")
        
        # Classify the large negative review
        if nmv_list[0] != '':
            ns_cv_n = TfidfVectorizer() #TF-IDF vectorizer for the large negative review
            feature_list = ns_cv_n.fit_transform(nmv_list)
            
            feature_list.resize(feature_list.shape[0],text_counts.shape[1])  # Resize the sparse matrix 'feature list' of the current sentnce to fit into the trained model
            
            classification = clf.predict(feature_list)
            neg_conf_num +=1 # Add one to the list of negative reviews generated
            if classification[0] == 1:
                print(" ~~~ Review 2 Classification: Positive")
            else:
                print(" ~~~ Review 2 Classification: Negative")
                neg_conf_metric += 1 # Correctly confirmed as a negative review; add one to the number of correct confirmations for negative reviews
        
        print ("\n")
        print(" # ___________________________________________________________ #")
        print(" # =========================================================== #")
        print(" # =========================================================== #")
        print(" # =========================================================== #")
        print(" # ___________________________________________________________ #")
        print ("\n")
        
        
        
        # ============================================================================= #
        # ============================================================================= #
        # ========== Markov Chain trained on positive and negative data sets ========== #        
        # ============================================================================= #
        # ============================================================================= #
        
        # Generate our small review segments
        print(" =================================================================== ")
        print(" ========== POSITIVE/NEGATIVE-REVIEW-TRAINED MARKOV CHAIN ========== ")
        print(" =================================================================== ")
        
        
        # Make the positive sentences and add them to a positive sentences list
        positive_sentences_list = []
        for i in range(review_num):
            new_review = ""
            for j in range(sentence_num):
                new_review += positive_text_model.make_sentence() + " "
            
            positive_sentences_list.append(new_review)
        
        # Make a TF-IDF vectorizer to convert our sentence into a list of features
        ns_cv_p2 = TfidfVectorizer()
        #feature_list = ns_cv_p2.fit_transform(positive_sentences_list)
        # Generate the feature list matrix based on the list of review fragments
        feature_list1 = tf.fit_transform(positive_sentences_list)
        feature_list1.resize(feature_list1.shape[0],text_counts.shape[1])  # Resize the sparse matrix 'feature list' of the current review fragment to fit into the trained model

        # Classify the fragments generated by the positively-trained Markov Chain
        classification_positive = clf.predict(feature_list1)
        
        
        # Make the positive sentences and add them to a negative sentences list
        negative_sentences_list = []
        for i in range(review_num):
            new_review = ""
            for j in range(sentence_num):
                new_review += negative_text_model.make_sentence() + " "
            
            negative_sentences_list.append(new_review)
        
        # Make a TF-IDF vectorizer to convert our sentence into a list of features
        ns_cv_n2 = TfidfVectorizer()
        #feature_list = ns_cv_n2.fit_transform(negative_sentences_list)
        # Generate the feature list matrix based on the list of review fragments
        feature_list2 = tf.fit_transform(negative_sentences_list)
        feature_list2.resize(feature_list2.shape[0],text_counts.shape[1])  # Resize the sparse matrix 'feature list' of the current review fragment to fit into the trained model
        
        # Classify the fragments generated by the negatively-trained Markov Chain
        classification_negative = clf.predict(feature_list2)
            
        
        # Lists to hold our positively- and negatively-classified review fragments
        positive_review_fragments_clss = []
        negative_review_fragments_clss = []
        
        # See how many of our positive review fragments were classified as positive
        positive_classifications = 0
        for i in range(review_num):
            if classification_positive[i] == 1:
                positive_review_fragments_clss.append(positive_sentences_list[i])
                positive_classifications +=1 # Add one to the number of sentences generated by the positively-trained Markov Chain that were classified as positive
            else:
                negative_review_fragments_clss.append(positive_sentences_list[i])
              
        # See the total number of positive-chain-generated review fragments classified as positive
        print("Number of positively-trained Markov Chain-generated review fragments classified as positive" + str(positive_classifications) + "/" + str(review_num))
        pos_class_metric.append(positive_classifications/review_num)
        
        # See how many of our negative review fragments were classified as positive
        negative_classifications = 0
        for i in range(review_num):
            if classification_negative[i] == 0:
                negative_review_fragments_clss.append(negative_sentences_list[i])
                negative_classifications +=1 # Add one to the number of sentences generated by the negatively-trained Markov Chain that were classified as negative
            else:
                positive_review_fragments_clss.append(negative_sentences_list[i])
                
        # See the total number of negative-chain-generated review fragments classified as negative
        print("Number of negatively-trained Markov Chain-generated review fragments classified as negative: " + str(negative_classifications) + "/" + str(review_num))
        neg_class_metric.append(negative_classifications/review_num)
        
        print("")
        
        # Print our first big positive review from only all the sentences generated positive-review-trained Markov Chain
        positive_movie_review_new = "" # Initialize the review
        for small_review in positive_sentences_list:
            positive_movie_review_new += small_review # Append each positive-chain-generated fragment to the positive review
            
        print(" >>> Long Review A (positive-chain-generated fragments):")
        print(positive_movie_review_new)
        pmv_list_new = [positive_movie_review_new] # Place the review into a list so that it can be vectorized by the TF-IDF vectorizer for classification
        review_list_grammar_check.append(positive_movie_review_new) # Append to the list of all reviews to check for grammar errors later
        print("")
        
        # Print our first big negative review from only all the sentences generated negative-review-trained Markov Chain
        negative_movie_review_new = "" # Initialize the review
        for small_review in negative_sentences_list:
            negative_movie_review_new += small_review # Append each negative-chain-generated fragment to the negative review
        
        print(" >>> Long Review B (negative-chain-generated fragments):")
        print(negative_movie_review_new)
        nmv_list_new = [negative_movie_review_new] # Place the review into a list so that it can be vectorized by the TF-IDF vectorizer for classification
        review_list_grammar_check.append(negative_movie_review_new) # Append to the list of all reviews to check for grammar errors later
        print("")
        
        
        # Classify the large positive review
        # Skip if empty
        if pmv_list_new[0] != '':
            ns_cv_p2new = TfidfVectorizer() # TF-IDF vectorizer for the large positive review
            #feature_list3 = ns_cv_p2new.fit_transform(pmv_list_new)
            feature_list3 = tf.fit_transform(pmv_list_new)
            
            feature_list3.resize(feature_list3.shape[0],text_counts.shape[1])  # Resize the sparse matrix 'feature list' of the current sentnce to fit into the trained model
            
            classification = clf.predict(feature_list3)
            if classification[0] == 1:
                print(" ~~~ Review A Classification: Positive")
            else:
                print(" ~~~ Review A Classification: Negative")
        
        # Classify the large negative review
        if nmv_list_new[0] != '':
            ns_cv_n2new = TfidfVectorizer() # TF-IDF vectorizer for the large negative review
            #feature_list4 = ns_cv_n2new.fit_transform(nmv_list_new)
            feature_list4 = tf.fit_transform(nmv_list_new)
            
            feature_list4.resize(feature_list4.shape[0],text_counts.shape[1])  # Resize the sparse matrix 'feature list' of the current sentnce to fit into the trained model
            
            classification = clf.predict(feature_list4)
            if classification[0] == 1:
                print(" ~~~ Review B Classification: Positive")
            else:
                print(" ~~~ Review B Classification: Negative")
        
        
        print("")
        print("==============================================")
        print("")
        
        # Create a new positive review based on the positively-classified fragments
        
        # Print our new big positive review based on the positively-classified sentences from both chains
        positive_movie_review_new_clss = "" # Initialize the review
        for small_review in positive_review_fragments_clss:
            positive_movie_review_new_clss += small_review # Append each positively-classified fragment to the positive review
        
        print(" >>> Review C (positively-classified fragments):")
        print(positive_movie_review_new_clss)
        pmv_list_new_clss = [positive_movie_review_new_clss] # Place the review into a list so that it can be vectorized by the TF-IDF vectorizer for classification
        review_list_grammar_check.append(positive_movie_review_new_clss) # Append to the list of all reviews to check for grammar errors later
        print("")
        
        # Create a new negative review based on the positively-classified fragments
        
        # Print our new big negative review based on the negatively-classified sentences from both chains
        negative_movie_review_new_clss = "" # Initialize the review
        for small_review in negative_review_fragments_clss:
            negative_movie_review_new_clss += small_review # Append each negatively-classified fragment to the negative review
        
        print(" >>> Review D (negatively-classified fragments):")
        print(negative_movie_review_new_clss)
        nmv_list_new_clss = [negative_movie_review_new_clss] # Place the review into a list so that it can be vectorized by the TF-IDF vectorizer for classification
        review_list_grammar_check.append(negative_movie_review_new_clss) # Append to the list of all reviews to check for grammar errors later
        print("")
        
        
        # Classify the new large positive review
        # Skip if empty
        if nmv_list_new_clss[0] != '':
            ns_cv_p2new_clss = TfidfVectorizer()
            #feature_list3 = ns_cv_p2new_clss.fit_transform(pmv_list_new)
            feature_list5 = tf.fit_transform(pmv_list_new)
            
            feature_list5.resize(feature_list5.shape[0],text_counts.shape[1])  # Resize the sparse matrix 'feature list' of the current review fragment to fit into the trained model
            
            classification = clf.predict(feature_list3)
            pos_conf_num_split +=1 # Add one to the list of positive reviews generated from the split chains
            if classification[0] == 1:
                print(" ~~~ Review C Classification: Positive")
                pos_conf_metric_split += 1  # Correctly confirmed as a positive review; add one to the number of correct confirmations for positive reviews
            else:
                print(" ~~~ Review C Classification: Negative")
        
        # Classify the new large negative review
        if nmv_list_new_clss[0] != '':
            ns_cv_n2new_clss =TfidfVectorizer()
            #feature_list4 = ns_cv_p2new_clss.fit_transform(nmv_list_new)
            feature_list6 = tf.fit_transform(nmv_list_new)
            
            feature_list6.resize(feature_list6.shape[0],text_counts.shape[1])  # Resize the sparse matrix 'feature list' of the current review fragment to fit into the trained model
            
            classification = clf.predict(feature_list4)
            neg_conf_num_split +=1 # Add one to the list of negative reviews generated from the split chains
            if classification[0] == 1:
                print(" ~~~ Review D Classification: Positive")
            else:
                print(" ~~~ Review D Classification: Negative")
                neg_conf_metric_split += 1  # Correctly confirmed as a negative review; add one to the number of correct confirmations for negative reviews
        print("\n\n\n\n\n\n\n\n\n\n")
                
                
    # Print our final output metrics
    print("\n\n\n\n\n\n\n\n\n\n")
    print("")
    print("===============================================================================================")
    print("===============================================================================================")
    print("===============================================================================================")
    print("\n >>>>> Final Metrics: \n")
    pos_class_percentage = (sum(pos_class_metric)/len(pos_class_metric)) * 100
    print("Positively-Trained Markov Chain --- Percentage of statements classified as positive: " + str(pos_class_percentage))
    neg_class_percentage = (sum(neg_class_metric)/len(neg_class_metric)) * 100
    print("Negatively-Trained Markov Chain --- Percentage of statements classified as negative: " + str(neg_class_percentage))
    print("Average positively-trained/negatively-trained statement classification 'accuracy': " + str((pos_class_percentage + neg_class_percentage)/2) )
    print("")
    pos_conf_percentage = (pos_conf_metric/pos_conf_num) * 100
    print("Positively-classified review confirmation accuracy (whole data set): " + str(pos_conf_percentage))
    neg_conf_percentage = (neg_conf_metric/neg_conf_num) * 100
    print("Negatively-classified review confirmation accuracy (whole data set): " + str(neg_conf_percentage))
    print("Average review confirmation accuracy (whole data set): " + str((pos_conf_percentage + neg_conf_percentage)/2))
    pos_conf_percentage_split = (pos_conf_metric_split/pos_conf_num_split) * 100
    print("Positively-classified review confirmation accuracy (split data set): " + str(pos_conf_percentage_split))
    neg_conf_percentage_split = (neg_conf_metric_split/neg_conf_num_split) * 100
    print("Negatively-classified review confirmation accuracy (split data set): " + str(neg_conf_percentage_split))
    
    
    for i in range(len(review_list_grammar_check)):
        grammar_err_gen_single = lang_tool.check(review_list_grammar_check[i])
        grammar_err_generated += grammar_err_gen_single
    print("Average grammar errors/review from reviews generated this run :" + str(grammar_err_generated/len(review_list_grammar_check)))
    
    
    sys.stdout = original_stdout # Reset the standard output to its original value
    

# We're finished!
print("")
print(" =====> Done!")


# Plot our results
# Positively-Trained Markov Chain - Percentage of statements classified as positive
plt.plot(pos_class_metric)
plt.xlabel('Run')
plt.ylabel('Percentage',rotation=0)
plt.title('Positively-Trained Markov Chain - Percentage of statements classified as positive:')
plt.show()
plt.savefig('positive_chain_graph.png')

# Negatively-Trained Markov Chain - Percentage of statements classified as negative
plt.plot(neg_class_metric)
plt.xlabel('Run')
plt.ylabel('Percentage',rotation=0)
plt.title('Negatively-Trained Markov Chain - Percentage of statements classified as negative:')
plt.show()
plt.savefig('negative_chain_graph.png')

