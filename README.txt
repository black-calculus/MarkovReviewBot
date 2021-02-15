***The Kaggle dataset's sentiment column is classified by strings ('positive' and 'negative'), so we simply quantized them into 0 and 1 respectively
 - Dataset link: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

SCRIPTS DETAILING:
newbot_tfidf.py - Our movie review generation and classification bot - the bulk of the code - generates our 600 reviews)
 ~ Will require the 'language_tool_python', 'markovify', 'sklearn', 'numpy', and 'pandas' packages
 ~ You may need to shut down Python manually (e.g. via Windows Task Manager) after running the script to release the large amount of memory taken up by the dataframe; memory deallocation handling was not figured out at the time of submission
grammar_err.py - Script to count the average grammar errors in the dataset (randomly samples 600 reviews from the dataset)
reviewproc.py - Script to calculate the average grammar errors in the generated reviews