from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']

llm = OpenAI(api_token=API_KEY)
pandas_ai = PandasAI(llm)

st.title("Prompt driven analysis with PandasAI")

# Specify the file path
file_path = "datasets.csv"

# Read CSV file
df = pd.read_csv(file_path)

st.write(df)

if os.path.isfile("temp_chart.png"):
    os.remove("temp_chart.png")

prompt = st.text_area("Enter your prompt:")

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response..."):
            response = pandas_ai.run(df, prompt=prompt)
            st.write(response)
            matplotlib.use('svg')
            if os.path.isfile("temp_chart.png"):
                st.image("temp_chart.png")

