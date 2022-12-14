"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib
import os
import pickle 

# app requirements
#from collections import defaultdict, namedtuple
#from htbuilder import div, big, h2, styles
#from htbuilder.units import rem
#from math import floor
#from textblob import TextBlob
#import altair as alt
#import datetime
#import functools
import pandas as pd
#import re
#import secrets_beta
import time
#import tweepy

# Data dependencies
import pandas as pd

# extract path information for raw data
from pathlib import Path

# Vectorizer
#tweet_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_vectorizer = open("mlr_vectorizer2.pkl", "rb")
tweet_cv = joblib.load(tweet_vectorizer) # loading your vectorizer from the pkl file 

# path information
dir_path = Path(__file__).parent
print(dir_path)

# Load your raw data
raw = pd.read_csv(dir_path / "train.csv") 

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.set_page_config(page_icon="🐤", page_title="Twitter Sentiment Analyzer")

	st.title("Twitter Sentiment Analyzer") #st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information", "Sentiment Classification", "Model Information", "Prediction", "Raw Data"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the Information page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Many companies would like to determine how their customers perceive climate change and whether or not they believe it is a real threat.")
		st.markdown("Knowledge of this would add to their market research efforts in gauging how their product/service may be received.")
		st.markdown("An accurate and robust solution to this problem would give the companies access to a broad understanding of consumer sentiment.")
		st.markdown("Sentiment that spans multiple demographic and geographics.")
		st.markdown("As a result it increases a companies' insights and informing future marketing strategies for the betterment of consumer experience and company performance.")
	
	# Building out the Sentiment Classification page
	if selection == "Sentiment Classification":
		st.info("General Information")
		st.markdown("Building a Machine Learning model that is able to classify whether or not a person believes in climate change, based on their novel tweet data. The tweets belonging to any of the following class descriptions:")
		st.markdown("2 News: the tweet links to factual news about climate change.")
		st.markdown("1 Pro: the tweet supports the belief of man-made climate change.")
		st.markdown("0 Neutral: the tweet neither supports nor refutes the belief of man-made climate change.")
		st.markdown("-1 Anti: the tweet does not believe in man-made climate change Variable definitions")

	# Building out the Model Information page
	if selection == "Model Information":
		st.info("Model Information")
		st.markdown("Multiple types of classification models have been used.")
		st.markdown("A model can be known as a classifier.")
		st.markdown("")
		st.markdown("1. Logistic Regression")
		st.markdown("Applies the sigmoid(logistic) function to keep the outcome a probabilistic value between 0 and 1.")
		st.markdown("")
		st.markdown("2. K-Nearest Neighbour.")
		st.markdown("Uses an algorithm that implements classification based on voting by the nearest k-neighbours of the target point.")
		st.markdown("")
		st.markdown("3. SVM")
		st.markdown("Categorizes the data with the separator drawn as a hyperplane.")
		st.markdown("")
		st.markdown("4. Decision Tree")
		st.markdown("Uses a tree structure and split points to classify the data.")
		st.markdown("")
		st.markdown("5. Random Forest")
		st.markdown("Constructs a multitude of decision trees during training.")
		st.markdown("")
		st.markdown("6. AdaBoost")
		st.markdown("Builds the model on the training data and then a second model to fix errors present in the first model by re-assigning weights to each instance, with higher weights assigned to incorrectly classified instances.")
	
	# Building out the Raw Data page
	if selection == "Raw Data":
		st.info("Raw Twitter Data")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the Predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Tweet Text Here (limited to 280 characters)","Typing...", max_chars=280)

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("mlr_model2.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
