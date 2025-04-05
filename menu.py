import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np

# ✅ Streamlit page config
st.set_page_config(page_title="Mess Forecasting Assistant", layout="centered")

# ✅ Load env vars and Gemini API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Load dataset
df = pd.read_csv("C:/Users/Admin/Desktop/NSUT1/Projects/Data/mess_data_5000_days.csv")
df['date'] = pd.to_datetime(df['date'])

# ✅ Prepare ANN data
df['day'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Encode menu if needed
if df['menu'].dtype == object:
    df['menu_encoded'] = df['menu'].astype('category').cat.codes

# ✅ Use only existing features (remove missing ones)
features = ['day', 'month', 'menu_encoded']  # removed is_holiday, temperature
target = ['attendees']

scaler = MinMaxScaler()
X = scaler.fit_transform(df[features])
y = df[target].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define ANN model in PyTorch
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(len(features), 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)

@st.cache_resource
def train_ann_model():
    model = ANNModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(300):
        model.train()
        outputs = model(X_tensor).view(-1, 1)
        loss = criterion(outputs, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

ann_model = train_ann_model()

# ✅ Gemini system prompt
SYSTEM_PROMPT = """
You are a mess forecasting assistant.
Based on historical data, predict the number of attendees today or on a given day.
Use context like date, day, menu to explain the forecast.
Return short, smart, and insightful explanations. Use emojis like 📊 🍛 ☀️ and markdown.
"""

# Extract date
def extract_date_from_text(text):
    try:
        for line in text.split('\n'):
            if line.lower().startswith("date:"):
                return pd.to_datetime(line.split(":")[1].strip()).date()
    except:
        return None

# Translate
def translate_response(text, target_language):
    if target_language == "English":
        return text
    model_translate = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
    translation_prompt = f"""
Translate the following forecast explanation into {target_language}. Keep the emojis and markdown formatting as it is.

{text}
"""
    translation_response = model_translate.generate_content(translation_prompt)
    return translation_response.text

# Forecast using ANN + Gemini explanation
def forecast_with_ann_and_gemini(user_query: str, target_lang: str):
    model_gemini = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

    # Step 1: Extract structured info
    context_prompt = f"""
{SYSTEM_PROMPT}

User: {user_query}

Extract these fields if possible:
- Date or day
- Menu item

Return in this format:
Date: ...
Menu: ...
"""
    response = model_gemini.generate_content(context_prompt)
    structured_info = response.text
    date = extract_date_from_text(structured_info) or datetime.now().date()

    # Simulate input features for prediction (manual values)
    input_dict = {
        'day': date.weekday(),
        'month': date.month,
        'menu_encoded': 1,  # default or use mapping later
    }

    input_data = scaler.transform(pd.DataFrame([input_dict]))
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    ann_model.eval()
    with torch.no_grad():
        predicted = ann_model(input_tensor).item()
        predicted = max(0, int(predicted))  # clamp to non-negative

    # Step 3: LLM summarization
    final_prompt = f"""
{SYSTEM_PROMPT}

User asked: {user_query}

Context:
{structured_info}

Forecast: Around {predicted} students are expected today.

Explain the reasoning like a professional data analyst:
- Mention possible factors
- Use markdown with emojis
- End with "Top influencing factors" as bullet points
"""
    final_response = model_gemini.generate_content(final_prompt)
    translated = translate_response(final_response.text, target_lang)
    return translated, predicted

# ✅ Streamlit UI
st.title("🍽️ Mess Forecasting Assistant (ANN + Gemini)")

language = st.selectbox("🌐 Select your preferred language:", ["English", "Hindi", "Spanish", "French", "German"])
st.markdown("Ask your assistant anything about food mess demand prediction.")

user_input = st.text_input("Ask your question:", placeholder="e.g. How many students if we serve Biryani on Sunday?")

if st.button("🔮 Generate Forecast"):
    if user_input.strip() != "":
        with st.spinner("Thinking..."):
            response, predicted = forecast_with_ann_and_gemini(user_input, language)
            st.success("Here's the forecast:")
            st.markdown(response)

            # Plot (basic bar)
            st.bar_chart(pd.DataFrame({"Predicted Attendance": [predicted]}))

            # Download
            buffer = StringIO()
            buffer.write(response)
            st.download_button("📅 Download Forecast Report", buffer.getvalue(), file_name="forecast_report.txt")
    else:
        st.warning("Please enter a question before generating forecast.")
