import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ✅ Streamlit UI setup
st.set_page_config(page_title="Nutrition Assistant", layout="centered")

# ✅ Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Load dataset
df = pd.read_csv("C:/Users/Admin/Desktop/NSUT1/Projects/Data/mess_data_5000_days.csv")
df['date'] = pd.to_datetime(df['date'])

# ✅ Load or simulate menu
menu_items = {
    "Biryani": {"calories": 480, "protein": 12, "fiber": 3},
    "Rajma": {"calories": 220, "protein": 9, "fiber": 6},
    "Salad": {"calories": 70, "protein": 2, "fiber": 4},
    "Paneer": {"calories": 300, "protein": 18, "fiber": 1},
    "Fried Rice": {"calories": 420, "protein": 7, "fiber": 2},
    "Daal": {"calories": 150, "protein": 10, "fiber": 5},
    "Roti": {"calories": 100, "protein": 3, "fiber": 2},
}

# ✅ System Prompt
SYSTEM_PROMPT = """
You are a smart canteen assistant AI 🍽️.
Suggest healthy meal combinations based on protein, fiber, and calories.
Use emojis (💪 🥗 🍛) and markdown format.
Always end with 🎉 You earned 10 Green Points for choosing a healthy meal!
"""

# ✅ Translator
def translate_response(text, target_language):
    if target_language == "English":
        return text
    translator = genai.GenerativeModel("models/gemini-2.0-flash-latest")
    prompt = f"Translate this text to {target_language}, preserve emojis and markdown:\n\n{text}"
    return translator.generate_content(prompt).text

# ✅ Nutrition recommendation
def recommend_nutritious_meal(query):
    model_nutrition = genai.GenerativeModel("models/gemini-1.5-pro-latest")

    menu_summary = "\n".join([f"- {item}: {info['calories']} cal, {info['protein']}g protein, {info['fiber']}g fiber"
                              for item, info in menu_items.items()])

    prompt = f"""
{SYSTEM_PROMPT}

Here is the mess menu:

{menu_summary}

User asked: "{query}"

Suggest 1 or 2 healthy combinations. Prefer high protein and high fiber, moderate calories.
Also say: "🎉 You earned 10 Green Points for choosing a healthy meal!"
"""
    response = model_nutrition.generate_content(prompt)
    return response.text

# ✅ Streamlit UI
st.title("🥗 Smart Nutrition Assistant (GenAI)")

language = st.selectbox("🌐 Select your language", ["English", "Hindi", "French", "German", "Spanish"])

st.markdown("Ask your assistant for a healthy meal recommendation!")

user_input = st.text_input(
    "Ask a question:", 
    placeholder="e.g. What should I eat today for a healthy meal?"
)

if st.button("🍱 Get Meal Suggestion"):
    if user_input.strip() != "":
        with st.spinner("Thinking..."):
            suggestion = recommend_nutritious_meal(user_input)
            translated = translate_response(suggestion, language)
            st.success("✅ Nutritional Recommendation:")
            st.markdown(translated)
    else:
        st.warning("Please enter a question to continue.")
