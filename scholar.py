import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
import re
from io import StringIO

# Set your Gemini API key from environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# System prompt asking only for a markdown table
SYSTEM_PROMPT = (
    "You are a scholarship assistant. Generate a list of scholarships based on the user's request.\n"
    "Return ONLY a markdown table with the following columns:\n"
    "Name | Provider | Amount | Deadline | Eligibility\n\n"
    "**Do not include any explanation, JSON, or text outside the table. Just give the markdown table.**"
)

# Streamlit App UI
st.set_page_config(page_title="🎓 Automated Scholarship Finder")
st.title("🎓 AI-Powered Scholarship Finder")

# Sidebar Filters
st.sidebar.header("Search Filters")
country = st.sidebar.text_input("Country", "India")
level = st.sidebar.selectbox("Education Level", ["Undergraduate", "Postgraduate", "PhD"])
field = st.sidebar.text_input("Field of Study", "STEM")
count = st.sidebar.slider("Number of Scholarships", 1, 10, 5)

# On click: Generate scholarships
if st.button("🔍 Find Scholarships"):
    with st.spinner("Finding scholarships..."):

        # Build user prompt
        user_prompt = (
            f"Generate a list of {count} scholarships for {level} students in {country} "
            f"studying in the field of {field}."
        )

        # Call Gemini
        chat = model.start_chat(history=[])
        response = chat.send_message(f"{SYSTEM_PROMPT}\n\n{user_prompt}", stream=True)

        # Collect streamed response
        full_response = ""
        for chunk in response:
            full_response += chunk.text

        # Try extracting the markdown table
        try:
            # Find the markdown table using regex
            table_match = re.search(r"\|.*\|", full_response, re.DOTALL)
            if not table_match:
                raise ValueError("No markdown table found in the response.")

            markdown_table = table_match.group(0)

            # Convert markdown table to DataFrame
            df = pd.read_csv(StringIO(markdown_table), sep="|", engine='python', skipinitialspace=True)

            # Clean up: remove empty and unnamed columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df = df.dropna(how="all")

            # Display final table
            st.subheader("📋 Matching Scholarships")
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error("⚠️ Failed to extract a valid table from Gemini's response.")
            st.exception(e)
