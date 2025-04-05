# Smart Mess Feedback System

import streamlit as st
import pandas as pd
import datetime
import sqlite3

# ------------------------------
# 1. DB Setup
# ------------------------------
conn = sqlite3.connect("mess_feedback.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS feedback
             (date TEXT, meal TEXT, taste TEXT, waste TEXT, comment TEXT)''')
conn.commit()

# ------------------------------
# 2. Title
# ------------------------------
st.set_page_config(page_title="Smart Mess Feedback System", layout="centered")
st.title("🍽️ Smart Mess Feedback System")

# ------------------------------
# 3. Feedback Form
# ------------------------------
st.subheader("📝 Give Your Feedback")

with st.form("feedback_form"):
    today = datetime.date.today().strftime("%Y-%m-%d")
    meal = st.selectbox("Which meal are you giving feedback for?", ["Breakfast", "Lunch", "Dinner"])
    taste = st.radio("How was the food?", ["😋 Great", "🙂 Okay", "😐 Average", "🤢 Not Good"])
    waste = st.radio("How much did you waste?", ["🔘 None", "🔘 Some", "🔘 A lot"])
    comment = st.text_area("Any suggestions or comments?")
    submit = st.form_submit_button("✅ Submit Feedback")

    if submit:
        c.execute("INSERT INTO feedback VALUES (?, ?, ?, ?, ?)",
                  (today, meal, taste, waste, comment))
        conn.commit()
        st.success("Thanks for your feedback! 🎉 You earned 10 Green Points!")

# ------------------------------
# 4. Dashboard Section
# ------------------------------
st.markdown("---")
st.subheader("📊 Feedback Dashboard")

if st.checkbox("Show Today's Feedback"):
    df_today = pd.read_sql_query(f"SELECT * FROM feedback WHERE date = '{today}'", conn)
    st.dataframe(df_today)

if st.checkbox("Show Weekly Summary"):
    df = pd.read_sql_query("SELECT * FROM feedback", conn)
    df['date'] = pd.to_datetime(df['date'])
    last_7 = df[df['date'] >= (pd.Timestamp.today() - pd.Timedelta(days=7))]

    st.write("### 🥗 Taste Summary")
    st.bar_chart(last_7['taste'].value_counts())

    st.write("### 🚮 Waste Summary")
    st.bar_chart(last_7['waste'].value_counts())

    st.write("### 🗣️ Common Feedback Comments")
    comments = last_7['comment'].dropna().tolist()
    for comment in comments:
        st.markdown(f"- {comment}")

# ------------------------------
# 5. Admin Section (Optional)
# ------------------------------
st.markdown("---")
st.subheader("🔐 Admin Insights")
if st.checkbox("Show Full Data"):
    full_df = pd.read_sql_query("SELECT * FROM feedback", conn)
    st.dataframe(full_df)

    # Download button
    csv = full_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Feedback CSV",
        data=csv,
        file_name='mess_feedback_data.csv',
        mime='text/csv',
    )

# ------------------------------
# END
# ------------------------------
