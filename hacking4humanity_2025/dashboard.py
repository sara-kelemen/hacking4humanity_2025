import streamlit as st
import instaloader
import pandas as pd
import torch
from transformers import pipeline
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Initialize Instaloader with Login Support
L = instaloader.Instaloader()

# Load session if available, otherwise login
def login_instagram(username, password):
    """Logs into Instagram and saves session to avoid repeated logins."""
    try:
        L.load_session_from_file(username)  # Load existing session
        st.success("✅ Session Loaded Successfully")
    except FileNotFoundError:
        try:
            L.login(username, password)  # Login with credentials
            L.save_session_to_file()  # Save session for future use
            st.success("✅ Logged in and Session Saved")
        except Exception as e:
            st.error(f"⚠️ Login Failed: {e}")

# Sentiment Analysis Model
sentiment_pipeline = pipeline("sentiment-analysis")

def sentiment_analysis(text):
    """Classifies text as Positive, Neutral, or Negative."""
    if text:
        result = sentiment_pipeline(text)[0] 
        return result['label'], round(result['score'], 2)
    return "Neutral", 0.0

# Function to get Instagram profile details
def get_instagram_profile(username: str):
    """Fetches public Instagram user profile details."""
    try:
        profile = instaloader.Profile.from_username(L.context, username)
        profile_data = {
            "Username": profile.username,
            "Full Name": profile.full_name,
            "Followers": profile.followers,
            "Following": profile.followees,
            "Bio": profile.biography,
            "Posts Count": profile.mediacount
        }
        return profile_data
    except Exception as e:
        st.error(f"Error fetching profile: {e}")
        return None

# Function to get Instagram posts
def get_instagram_posts(username, max_posts=10):
    """Fetches most recent Instagram posts."""
    try:
        profile = instaloader.Profile.from_username(L.context, username)
        posts_data = []

        for post in profile.get_posts():
            posts_data.append({
                "Date": post.date_utc,
                "Caption": post.caption,
                "Likes": post.likes,
                "Comments": post.comments,
                "URL": post.url
            })
            if len(posts_data) >= max_posts:
                break
            time.sleep(5)  # Delay to avoid rate limits

        return posts_data
    except Exception as e:
        st.error(f"Error fetching posts: {e}")
        return None
import time

def get_instagram_posts(username, max_posts=10):
    """Fetch Instagram posts with rate limiting."""
    try:
        profile = instaloader.Profile.from_username(L.context, username)
        posts_data = []

        for post in profile.get_posts():
            posts_data.append({
                "Date": post.date_utc,
                "Caption": post.caption,
                "Likes": post.likes,
                "Comments": post.comments,
                "URL": post.url
            })
            if len(posts_data) >= max_posts:
                break
            
            time.sleep(10)  # **Wait 10 seconds between each request**

        return posts_data
    except Exception as e:
        st.error(f"Error fetching posts: {e}")
        return None


#  **STREAMLIT UI**
st.title("📊 Instagram Sentiment Dashboard")

# 1**Login Input**
st.subheader("🔑 Instagram Login")
instagram_username = st.text_input("Enter your Instagram Username:")
instagram_password = st.text_input("Enter your Instagram Password:", type="password")

if st.button("Login"):
    if instagram_username and instagram_password:
        login_instagram(instagram_username, instagram_password)
    else:
        st.warning("⚠️ Please enter both username and password.")

# **User Input for Instagram Profile**
username = st.text_input("Reenter Instagram Username to Analyze:")

#  **Fetch and Display Data When User Submits**
if username:
    st.subheader(f"📌 Profile Overview: {username}")

    profile_info = get_instagram_profile(username)
    if profile_info:
        st.write(profile_info)

    st.subheader("📊 Sentiment Analysis of Instagram Posts & Comments")

    posts = get_instagram_posts(username, max_posts=10)

    if posts:
        df = pd.DataFrame(posts)

        df["Comment Sentiments"] = df["Comments Sentiments"].apply(
            lambda x: ", ".join([f"{s} ({c})" for s, c in x]) if x else "No Comments" #  creates a formatted string for each tuple (s, c), where s and c are the first and second elements of the tuple
        )

        st.dataframe(df[["Date", "Caption", "Caption Sentiment", "Caption Confidence", "Comment Sentiments"]])

        # **Sentiment Distribution in Comments**
        st.subheader("📊 Sentiment Distribution in Comments")

        all_comment_sentiments = [s for post in df["Comments Sentiments"] for s, _ in post]

        if all_comment_sentiments:
            sentiment_counts = pd.Series(all_comment_sentiments).value_counts()
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind="bar", color=["green", "gray", "red"], ax=ax)
            plt.xlabel("Sentiment Category")
            plt.ylabel("Number of Comments")
            plt.title("Sentiment Breakdown in Comments")
            st.pyplot(fig)

        # **Sentiment Trend Over Time**
        st.subheader("📈 Comment Sentiment Trend Over Time")

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        df["Avg Comment Confidence"] = df["Comments Sentiments"].apply(
            lambda x: np.mean([c for _, c in x]) if x else 0 # extracts the second element (c) from each tuple in 
        )

        fig, ax = plt.subplots()
        sns.lineplot(data=df, x="Date", y="Avg Comment Confidence", marker="o", ax=ax)
        plt.xticks(rotation=45)
        plt.title("Average Comment Sentiment Confidence Over Time")
        st.pyplot(fig)

