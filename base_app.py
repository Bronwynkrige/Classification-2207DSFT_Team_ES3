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
from collections import defaultdict, namedtuple
from htbuilder import div, big, h2, styles
from htbuilder.units import rem
from math import floor
from textblob import TextBlob
import altair as alt
import datetime
import functools
import pandas as pd
import re
#import secrets_beta
import time
import tweepy

# Data dependencies
import pandas as pd

# extract path information for raw data
from pathlib import Path

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

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

	st.write('<base target="_blank">', unsafe_allow_html=True)

	prev_time = [time.time()]

	a, b = st.columns([1, 10])

	with a:
		st.text("")
		st.image("logoOfficial.png", width=50)
	with b:
		st.title("Twitter Sentiment Classifier")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Many companies would like to determine how their customers perceive climate change and whether or not they believe it is a real threat.")
		st.markdown("Knowledge of this would add to their market research efforts in gauging how their product/service may be received.")
		st.markdown("An accurate and robust solution to this problem would give the companies access to a broad understanding of consumer sentiment.")
		st.markdown("Sentiment that spans multiple demographic and geographics.")
		st.markdown("As a result it increases a companies' insights and informing future marketing strategies for the betterment of consumer experience and company performance.")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
