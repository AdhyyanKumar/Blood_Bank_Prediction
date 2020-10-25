import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import random
import matplotlib.pyplot as plt
from datetime import datetime
import time
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
random.seed(50)
Blood_DataSetM = { 
    "January": [int(9758369/random.uniform(11.25,12)),int(8552451/random.uniform(11.25,12)),int(9477572/random.uniform(11.25,12)),int(9144194/random.uniform(11.25,12)),int(9128393/random.uniform(11.25,12)),int(9217026/random.uniform(11.25,12)),int(9573158/random.uniform(11.25,12)),int(10386123/random.uniform(11.25,12)),int(11553736/random.uniform(11.25,12)),int(12354785/random.uniform(11.25,12)),int(13388323/random.uniform(11.25,12))], 
    "February": [int(9758369/random.uniform(11.9,12.1)),int(8552451/random.uniform(11.9,12.1)),int(9477572/random.uniform(11.9,12.1)),int(9144194/random.uniform(11.9,12.1)),int(9128393/random.uniform(11.9,12.1)),int(9217026/random.uniform(11.9,12.1)),int(9573158/random.uniform(11.9,12.1)),int(10386123/random.uniform(11.9,12.1)),int(11553736/random.uniform(11.9,12.1)),int(12354785/random.uniform(11.9,12.1)),int(13388323/random.uniform(11.9,12.1))], 
    "March": [int(9758369/random.uniform(12,12.75)),int(8552451/random.uniform(12,12.75)),int(9477572/random.uniform(12,12.75)),int(9144194/random.uniform(12,12.75)),int(9128393/random.uniform(12,12.75)),int(9217026/random.uniform(12,12.75)),int(9573158/random.uniform(12,12.75)),int(10386123/random.uniform(12,12.75)),int(11553736/random.uniform(12,12.75)),int(12354785/random.uniform(12,12.75)),int(13388323/random.uniform(12,12.75))], 
    "April": [int(9758369/random.uniform(11.9,12.1)),int(8552451/random.uniform(11.9,12.1)),int(9477572/random.uniform(11.9,12.1)),int(9144194/random.uniform(11.9,12.1)),int(9128393/random.uniform(11.9,12.1)),int(9217026/random.uniform(11.9,12.1)),int(9573158/random.uniform(11.9,12.1)),int(10386123/random.uniform(11.9,12.1)),int(11553736/random.uniform(11.9,12.1)),int(12354785/random.uniform(11.9,12.1)),int(13388323/random.uniform(11.9,12.1))], 
    "May": [int(9758369/random.uniform(11.25,12)),int(8552451/random.uniform(11.25,12)),int(9477572/random.uniform(11.25,12)),int(9144194/random.uniform(11.25,12)),int(9128393/random.uniform(11.25,12)),int(9217026/random.uniform(11.25,12)),int(9573158/random.uniform(11.25,12)),int(10386123/random.uniform(11.25,12)),int(11553736/random.uniform(11.25,12)),int(12354785/random.uniform(11.25,12)),int(13388323/random.uniform(11.25,12))], 
    "June": [int(9758369/random.uniform(12,12.75)),int(8552451/random.uniform(12,12.75)),int(9477572/random.uniform(12,12.75)),int(9144194/random.uniform(12,12.75)),int(9128393/random.uniform(12,12.75)),int(9217026/random.uniform(12,12.75)),int(9573158/random.uniform(12,12.75)),int(10386123/random.uniform(12,12.75)),int(11553736/random.uniform(12,12.75)),int(12354785/random.uniform(12,12.75)),int(13388323/random.uniform(12,12.75))], 
    "July": [int(9758369/random.uniform(11.9,12.1)),int(8552451/random.uniform(11.9,12.1)),int(9477572/random.uniform(11.9,12.1)),int(9144194/random.uniform(11.9,12.1)),int(9128393/random.uniform(11.9,12.1)),int(9217026/random.uniform(11.9,12.1)),int(9573158/random.uniform(11.9,12.1)),int(10386123/random.uniform(11.9,12.1)),int(11553736/random.uniform(11.9,12.1)),int(12354785/random.uniform(11.9,12.1)),int(13388323/random.uniform(11.9,12.1))], 
    "August": [int(9758369/random.uniform(11.25,12)),int(8552451/random.uniform(11.25,12)),int(9477572/random.uniform(11.25,12)),int(9144194/random.uniform(11.25,12)),int(9128393/random.uniform(11.25,12)),int(9217026/random.uniform(11.25,12)),int(9573158/random.uniform(11.25,12)),int(10386123/random.uniform(11.25,12)),int(11553736/random.uniform(11.25,12)),int(12354785/random.uniform(11.25,12)),int(13388323/random.uniform(11.25,12))], 
    "September": [int(9758369/random.uniform(12,12.75)),int(8552451/random.uniform(12,12.75)),int(9477572/random.uniform(12,12.75)),int(9144194/random.uniform(12,12.75)),int(9128393/random.uniform(12,12.75)),int(9217026/random.uniform(12,12.75)),int(9573158/random.uniform(12,12.75)),int(10386123/random.uniform(12,12.75)),int(11553736/random.uniform(12,12.75)),int(12354785/random.uniform(12,12.75)),int(13388323/random.uniform(12,12.75))], 
    "Octuber": [int(9758369/random.uniform(11.9,12.1)),int(8552451/random.uniform(11.9,12.1)),int(9477572/random.uniform(11.9,12.1)),int(9144194/random.uniform(11.9,12.1)),int(9128393/random.uniform(11.9,12.1)),int(9217026/random.uniform(11.9,12.1)),int(9573158/random.uniform(11.9,12.1)),int(10386123/random.uniform(11.9,12.1)),int(11553736/random.uniform(11.9,12.1)),int(12354785/random.uniform(11.9,12.1)),int(13388323/random.uniform(11.9,12.1))], 
    "November": [int(9758369/random.uniform(12,12.75)),int(8552451/random.uniform(12,12.75)),int(9477572/random.uniform(12,12.75)),int(9144194/random.uniform(12,12.75)),int(9128393/random.uniform(12,12.75)),int(9217026/random.uniform(12,12.75)),int(9573158/random.uniform(12,12.75)),int(10386123/random.uniform(12,12.75)),int(11553736/random.uniform(12,12.75)),int(12354785/random.uniform(12,12.75)),int(13388323/random.uniform(12,12.75))], 
    "December": [int(9758369/random.uniform(11.25,12)),int(8552451/random.uniform(11.25,12)),int(9477572/random.uniform(11.25,12)),int(9144194/random.uniform(11.25,12)),int(9128393/random.uniform(11.25,12)),int(9217026/random.uniform(11.25,12)),int(9573158/random.uniform(11.25,12)),int(10386123/random.uniform(11.25,12)),int(11553736/random.uniform(11.25,12)),int(12354785/random.uniform(11.25,12)),int(13388323/random.uniform(11.25,12))] 
}
Blood_UsageM = pd.DataFrame(Blood_DataSetM, index=[2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]) 
Blood_UsageM.to_csv("Blood_Usage.csv")
PAGE_CONFIG = {"page_title": "Blood Prediction.in", "page_icon": ":syringe:","layout": "centered"}
st.beta_set_page_config(**PAGE_CONFIG)
blabla = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''
#https://images.unsplash.com/photo-1542281286-9e0a16bb7366 url https://healthcare.utah.edu/healthfeed/postings/2014/images/factoid-blood-donation.jpg
st.markdown(blabla, unsafe_allow_html=True)
buttonColouringTool = '''
<style>
.stButton>button {
    color: #4F8BF9;
    border-radius: 50%;
    height: 4em;
    width: 4em;
}

.stTextInput>div>div>input {
    color: #4F8BF9;
}

.stNumberInput>div>input {
    color: #4F8BF9;
}
</style>
'''
st.markdown(buttonColouringTool, unsafe_allow_html=True)
value = 0
counter = 0
df = pd.read_csv("Blood_Usage.csv")
df.columns = ["Year", "January", "Feburary", "March", "April", "May", "June", "July", "August", "September", "Octuber", "November", "December"]
year = datetime.now().year
st.bgcolor = "#ffcccb"
bbp = """<div style="background: #ffcccb; padding: 50px;"><h1 style = "color: black; text-align: center;">Blood Bank Prediction</h1></div>"""
st.markdown(bbp,unsafe_allow_html=True)
if year not in df["Year"]:
  month = datetime.now().month
  m_c_m = """<div style="background: #add8e6; padding: 10px;"><h2 style = "color: black; text-align: center;">Do you want prediction for current month or choosing another month?</h2></div>"""
  #m_c_m_2 = """<div style="background: #add8e6; padding: 10px;"><h2 style = "color: black; text-align: center;">Select m for this month or select c for chosing another month.</h2></div>"""
  st.markdown(m_c_m, unsafe_allow_html=True)
  #st.markdown(m_c_m_2, unsafe_allow_html=True)
  m_c = st.selectbox('',("Select a option",'current','another'))
  if m_c:
    if m_c == "current":
      if month > 2:
        st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">For how many months do you have dataset for?</h3></div>""",unsafe_allow_html=True)
        for_month_got = st.selectbox(" ",("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
        if for_month_got:
          if for_month_got != "Select a option":
            noErrorThing = "    "
            df1 = df
            df1.columns = ["0","1","2","3","4","5","6","7","8","9","10","11","12"]
            if for_month_got == 1:
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
            elif for_month_got == 2:
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
            elif for_month_got == 3:
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
            elif for_month_got == 4:
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
            elif for_month_got == 5:
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got5 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d5 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
            elif for_month_got == 6:
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got5 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d5 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got6 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d6 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
            elif for_month_got == 7:
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got5 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d5 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got6 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d6 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got7 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d7 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
            elif for_month_got == 8:
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got5 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d5 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got6 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d6 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got7 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d7 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got8 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d8 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
            elif for_month_got == 9:
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got5 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d5 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got6 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d6 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got7 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d7 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got8 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d8 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got9 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d9 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
            elif for_month_got == 10:
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got5 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d5 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got6 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d6 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got7 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d7 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got8 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d8 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got9 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d9 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got10 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d10 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
            elif for_month_got == 11:
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got5 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d5 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got6 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d6 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got7 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d7 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got8 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d8 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got9 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d9 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got10 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d10 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got11 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d11 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
            elif for_month_got == 12:
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got5 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d5 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got6 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d6 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got7 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d7 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got8 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d8 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got9 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d9 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got10 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d10 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got11 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d11 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
              month_got12 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
              st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
              month_got_d12 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
              noErrorThing = noErrorThing + " "
            predicttion = st.button("Predict.")
            if predicttion:
              if for_month_got == 1:
                new_row = {
                 "0": year, str(month_got1): month_got_d1
                }
              elif for_month_got == 2:
                new_row = {
                  "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2
                }
              elif for_month_got == 3:
                new_row = {
                  "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3
                }
              elif for_month_got == 4:
                new_row = {
                  "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4
                }
              elif for_month_got == 5:
                new_row = {
                  "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4, str(month_got5): month_got_d5
                }
              elif for_month_got == 6:
                new_row = {
                  "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4, str(month_got5): month_got_d5, str(month_got6): month_got_d6
                }
              elif for_month_got == 7:
                new_row = {
                  "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4, str(month_got5): month_got_d5, str(month_got6): month_got_d6, str(month_got7): month_got_d7
                }
              elif for_month_got == 8:
                new_row = {
                  "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4, str(month_got5): month_got_d5, str(month_got6): month_got_d6, str(month_got7): month_got_d7, str(month_got8): month_got_d8
                }
              elif for_month_got == 9:
                new_row = {
                  "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4, str(month_got5): month_got_d5, str(month_got6): month_got_d6, str(month_got7): month_got_d7, str(month_got8): month_got_d8, str(month_got9): month_got_d9
                }
              elif for_month_got == 10:
                new_row = {
                  "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4, str(month_got5): month_got_d5, str(month_got6): month_got_d6, str(month_got7): month_got_d7, str(month_got8): month_got_d8, str(month_got9): month_got_d9, str(month_got10): month_got_d10
                }
              elif for_month_got == 11:
                new_row = {
                  "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4, str(month_got5): month_got_d5, str(month_got6): month_got_d6, str(month_got7): month_got_d7, str(month_got8): month_got_d8, str(month_got9): month_got_d9, str(month_got10): month_got_d10, str(month_got11): month_got_d11
                }
              elif for_month_got == 12:
                new_row = {
                  "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4, str(month_got5): month_got_d5, str(month_got6): month_got_d6, str(month_got7): month_got_d7, str(month_got8): month_got_d8, str(month_got9): month_got_d9, str(month_got10): month_got_d10, str(month_got11): month_got_d11, str(month_got12): month_got_d12
                }
              df = df.append(new_row, ignore_index=True)
              imputer = IterativeImputer()
              imputed_DF = pd.DataFrame(imputer.fit_transform(df))
              curMonth = imputed_DF[month][11]
              prediction = "In " + str(year) + ", this month ("+str(month)+") the blood required will be approximately " + str(int(curMonth)) + " litres."
              prediction1 = """<div style="background: 	#90EE90; padding: 10px;"><h4 style = "color: black; text-align: center;">"""+str(prediction)+"""</h4></div>"""
              st.markdown(prediction1, unsafe_allow_html=True)
              imputed_DF.columns = ["Year", "January", "Feburary", "March", "April", "May", "June", "July", "August", "September", "Octuber", "November", "December"]
      else:
        d = """<div style="background: #add8e6; padding: 10px;"><h4 style = "color: black; text-align: center;">What was the blood required in January prevouis year? </h4></div>"""
        st.markdown(d, unsafe_allow_html=True)
        value = int(st.text_input("", 0))
        e = """<div style="background: #add8e6; padding: 10px;"><h4 style = "color: black; text-align: center;">What was the blood required in Febuary prevouis year? </h4></div>"""
        st.markdown(e, unsafe_allow_html=True)
        value1 = int(st.text_input(" ", 0))
        submit = st.button('Predict.')
        if submit:
          new_row = {
              "Year": year-1, "January": value, "Feburary": value1
          }
          df = df.append(new_row, ignore_index=True)
          imputer = IterativeImputer()
          imputed_DF = pd.DataFrame(imputer.fit_transform(df))
          curMonth = imputed_DF[12][11]
          prediction = "In " + str(year) + ", this month ("+str(month)+") the blood required will be approximately " + str(int(curMonth)) + " litres."
          prediction1 = """<div style="background: 	#90EE90; padding: 10px;"><h4 style = "color: black; text-align: center;">"""+str(prediction)+"""</h4></div>"""
          st.markdown(prediction1, unsafe_allow_html=True)
          imputed_DF.columns = ["Year", "January", "Feburary", "March", "April", "May", "June", "July", "August", "September", "Octuber", "November", "December"]
    elif m_c == "another":
      st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Select the number of the month for which you want prediction.</h3></div>""",unsafe_allow_html=True)
      for_month = st.selectbox("",("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
      if for_month:
        if for_month != "Select a option":
          st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">For how many months do you have dataset for?</h3></div>""",unsafe_allow_html=True)
          for_month_got = st.selectbox(" ",("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
          if for_month_got:
            if for_month_got != "Select a option":
              noErrorThing = "    "
              df1 = df
              df1.columns = ["0","1","2","3","4","5","6","7","8","9","10","11","12"]
              if for_month_got == 1:
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
              elif for_month_got == 2:
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
              elif for_month_got == 3:
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
              elif for_month_got == 4:
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
              elif for_month_got == 5:
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got5 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d5 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
              elif for_month_got == 6:
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got5 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d5 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got6 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d6 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
              elif for_month_got == 7:
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got5 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d5 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got6 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d6 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got7 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d7 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
              elif for_month_got == 8:
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got5 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d5 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got6 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d6 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got7 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d7 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got8 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d8 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
              elif for_month_got == 9:
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got5 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d5 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got6 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d6 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got7 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d7 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got8 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d8 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got9 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d9 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
              elif for_month_got == 10:
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got5 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d5 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got6 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d6 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got7 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d7 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got8 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d8 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got9 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d9 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got10 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d10 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
              elif for_month_got == 11:
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got5 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d5 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got6 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d6 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got7 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d7 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got8 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d8 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got9 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d9 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got10 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d10 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got11 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d11 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
              elif for_month_got == 12:
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got1 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d1 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got2 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d2 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got3 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d3 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got4 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d4 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got5 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d5 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got6 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d6 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got7 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d7 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got8 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d8 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got9 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d9 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got10 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d10 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got11 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d11 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the month you have dataset for.</h3></div>""",unsafe_allow_html=True)
                month_got12 = st.selectbox(noErrorThing, ("Select a option",1,2,3,4,5,6,7,8,9,10,11,12))
                st.markdown("""<div style="background: #add8e6; padding: 10px;"><h3 style = "color: black; text-align: center;">Write the blood required for that month.</h3></div>""",unsafe_allow_html=True)
                month_got_d12 = st.number_input(noErrorThing, min_value=0, max_value=9007199254740991, value = 0, step=1)
                noErrorThing = noErrorThing + " "
              predicttion = st.button("Predict.")
              if predicttion:
                if for_month_got == 1:
                  new_row = {
                    "0": year, str(month_got1): month_got_d1
                  }
                elif for_month_got == 2:
                  new_row = {
                    "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2
                  }
                elif for_month_got == 3:
                  new_row = {
                    "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3
                  }
                elif for_month_got == 4:
                  new_row = {
                    "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4
                  }
                elif for_month_got == 5:
                  new_row = {
                    "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4, str(month_got5): month_got_d5
                  }
                elif for_month_got == 6:
                  new_row = {
                    "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4, str(month_got5): month_got_d5, str(month_got6): month_got_d6
                  }
                elif for_month_got == 7:
                  new_row = {
                    "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4, str(month_got5): month_got_d5, str(month_got6): month_got_d6, str(month_got7): month_got_d7
                  }
                elif for_month_got == 8:
                  new_row = {
                    "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4, str(month_got5): month_got_d5, str(month_got6): month_got_d6, str(month_got7): month_got_d7, str(month_got8): month_got_d8
                  }
                elif for_month_got == 9:
                  new_row = {
                    "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4, str(month_got5): month_got_d5, str(month_got6): month_got_d6, str(month_got7): month_got_d7, str(month_got8): month_got_d8, str(month_got9): month_got_d9
                  }
                elif for_month_got == 10:
                  new_row = {
                    "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4, str(month_got5): month_got_d5, str(month_got6): month_got_d6, str(month_got7): month_got_d7, str(month_got8): month_got_d8, str(month_got9): month_got_d9, str(month_got10): month_got_d10
                  }
                elif for_month_got == 11:
                  new_row = {
                    "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4, str(month_got5): month_got_d5, str(month_got6): month_got_d6, str(month_got7): month_got_d7, str(month_got8): month_got_d8, str(month_got9): month_got_d9, str(month_got10): month_got_d10, str(month_got11): month_got_d11
                  }
                elif for_month_got == 12:
                  new_row = {
                    "0": year, str(month_got1): month_got_d1, str(month_got2): month_got_d2, str(month_got3): month_got_d3, str(month_got4): month_got_d4, str(month_got5): month_got_d5, str(month_got6): month_got_d6, str(month_got7): month_got_d7, str(month_got8): month_got_d8, str(month_got9): month_got_d9, str(month_got10): month_got_d10, str(month_got11): month_got_d11, str(month_got12): month_got_d12
                  }
                df = df.append(new_row, ignore_index=True)
                imputer = IterativeImputer()
                imputed_DF = pd.DataFrame(imputer.fit_transform(df))
                curMonth = imputed_DF[for_month][11]
                prediction = "In " + str(year) + ", the chosen month ("+str(for_month)+") the blood required will be approximately " + str(int(curMonth)) + " litres."
                prediction1 = """<div style="background: 	#90EE90; padding: 10px;"><h4 style = "color: black; text-align: center;">"""+str(prediction)+"""</h4></div>"""
                st.markdown(prediction1, unsafe_allow_html=True)
                imputed_DF.columns = ["Year", "January", "Feburary", "March", "April", "May", "June", "July", "August", "September", "Octuber", "November", "December"]