import streamlit as st
import pandas as pd
import torch
from transformers import pipeline
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime, timedelta

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Mock Instagram profile data
mock_acc = {
    "username": "sarkelemen",
    "full_name": "Sara Kelemen",
    "followers": 999,
    "following": 122,
    "bio": "duq 2025",
    "posts_count": 10
}

# Mock post captions
captions = [
    "Happy new years", "Happy 21st birthday to my bestie", "I love summer", "Concert!",
    "I hate social media", "Meow", "Puppy pics", "Happy graduation", "WTFFF", "hahaha",
    "I cant wait to graduate", "Senior sunday", "Love love love you", "What even is happening",
    "I love my friends", "Me n my loverss", "<3"
]
unique_captions = random.sample(captions, min(len(captions), 10))  
# Generate mock Instagram posts
mock_posts = [
    {
        "id": f"post_{i}",
        "caption": unique_captions[i],
        "media_type": "IMAGE",
        "media_url": f"https://instagram.com/{random.randint(1000, 9999)}.jpg",
        "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
        "likes": random.randint(50, 500),
        "comments_count": random.randint(5, 50),
    }
    for i in range(len(unique_captions)) 
]

df_posts = pd.DataFrame(mock_posts)

# Personal Instagram-style comments with timestamps
personal_comments = [
    "This vacation was the best! 🌴☀️", "I can't believe how cute your dog is! 🐶",
    "Happy birthday! Hope you have a fantastic day! 🎉", "Ugh, today was so stressful. 😩",
    "Why does everything always go wrong for me? 😭", "That food looks delicious! 🤤 Where did you get it?",
    "Congratulations on your new job! You deserve it! 👏", "I miss you so much! We need to catch up soon. ❤️",
    "I can't believe this happened... so disappointed. 😞", "Such a fun night! Can't wait to do it again! 🍾",
    "This is so unfair! I hate dealing with this. 😡", "What an amazing sunset! Nature is incredible. 🌅",
    "LOL this made me laugh so hard. 😂", "Why do Mondays have to be like this? 😒",
    "So proud of you and all you've accomplished! 💪", "This is the cutest baby I've ever seen! 😍",
    "Feeling super anxious today. Hoping for a better tomorrow.", "OMG this is so nostalgic! Takes me back. 💭",
    "I need coffee... like, now. ☕😅", "I love spending time with my family. Nothing better! ❤️"
]

# Generate random timestamps for each comment
comment_dates = [(datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d") for _ in personal_comments]

# Apply sentiment analysis to each comment while associating timestamps
comment_sentiments = [
    {"comment": c, "timestamp": t, **sentiment_pipeline(c)[0]} 
    for c, t in zip(personal_comments, comment_dates)
]

# Convert to DataFrame
df_sentiments = pd.DataFrame(comment_sentiments)

# Convert timestamp to datetime for sorting
df_sentiments["timestamp"] = pd.to_datetime(df_sentiments["timestamp"])

# Group by date and compute sentiment trend over time
sentiment_trend = df_sentiments.groupby("timestamp")["label"].value_counts().unstack().fillna(0)


df_profile = pd.DataFrame({
    "Attribute": ["Full Name", "Posts", "Followers", "Following", "Bio"],
    "Value": [mock_acc["full_name"], mock_acc["posts_count"], mock_acc["followers"], mock_acc["following"], mock_acc["bio"]]
})



# Streamlit UI
st.title("📊 Instagram Sentiment Dashboard")
st.subheader(f"📌 Profile Overview: {mock_acc['username']}")
st.markdown(f"""
**👤 Full Name:** {mock_acc['full_name']}  
📸 **Posts:** {mock_acc['posts_count']}  
👥 **Followers:** {mock_acc['followers']}  
🔄 **Following:** {mock_acc['following']}  
📝 **Bio:** {mock_acc['bio']}  
""")

# Display recent posts
st.subheader("📸 Recent Post Data")
st.dataframe(df_posts)

# Display comments with sentiment analysis
st.subheader("💬 Comment Sentiment Analysis")
st.dataframe(df_sentiments)

# Sentiment distribution bar chart
st.subheader("📊 Sentiment Distribution of Instagram Comments")
sentiment_counts = df_sentiments["label"].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="coolwarm", ax=ax)
plt.xlabel("Sentiment Category")
plt.ylabel("Number of Comments")
plt.title("Sentiment Distribution of Instagram Comments")
st.pyplot(fig)

# Sentiment trend over time (Line Chart)
st.subheader("📈 Comment Sentiment Trend Over Time")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=sentiment_trend, markers=True, ax=ax)
plt.xlabel("Date")
plt.ylabel("Number of Comments")
plt.title("Sentiment Trend Over Time")
plt.xticks(rotation=45)
plt.legend(title="Sentiment")
st.pyplot(fig)

# NLP TO ADVISE USERS
text_generator = pipeline("text-generation", model="tiiuae/falcon-40b")

# Generate recommendations using Falcon-40B
def generate_recommendations_falcon():
    prompt = (
        "Generate a structured list of five actionable Instagram recommendations "
        "that encourage positive engagement and reduce negative online interactions. "
        "Each recommendation should be concise and formatted as a bullet point."
    )
    response = text_generator(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]
    
    # Format recommendations into bullets
    recommendations = "\n".join([f"🔹 {line.strip()}" for line in response.split(".") if line.strip()])
    return recommendations

st.subheader("🌱 AI-Generated Recommendations for Better Social Media Engagement")
st.markdown(generate_recommendations_falcon())