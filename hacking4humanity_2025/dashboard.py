import streamlit as st
import instaloader
import pandas as pd
import torch
from transformers import pipeline
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

def get_instagram_profile(username: str):
    """This will extract the instagram user's account information, if they are public.

    Args:
        username (str): username

    Returns:
        profile_data: Username, name of user, who are the followers, who they follow, biography, and amount of posts.
    """

    L = instaloader.Instaloader()
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

def get_instagram_posts(username, max_posts=10):
    """Takes the most recent posts from a user for sentiment analysis and trends over time.

    Args:
        username (_type_): username
        max_posts (int, optional): the maximum amount of posts the dashboard will analyze. Defaults to 10.

    Returns:
        posts_data: each post's date, likes, comments, caption, and URL
    """
    L = instaloader.Instaloader()
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

    return posts_data

def fetch_profile(username: str):
    """Function to grab username from user input.

    Args:
        username (str): username input

    Returns:
        profile: the profile to be analyzed by the dashboard
    """
    try:
        profile = get_instagram_profile(username)
        return profile
    except Exception as e:
        st.error(f"Error fetching profile: {e}")
        return None
    
def fetch_posts(username, max_posts=10):
    """Fetch Instagram posts and analyze sentiment in captions and comments."""
    try:
        posts = get_instagram_posts(username, max_posts)

        for post in posts:
            # Sentiment Analysis for Captions
            sentiment, confidence = sentiment_analysis(post["Caption"])
            post["Caption Sentiment"] = sentiment
            post["Caption Confidence"] = confidence
            
            # Analyze Comments (if available)
            if post["Comments"] > 0:
                comment_sentiments = []
                for _ in range(post["Comments"]):  # Simulating comment extraction
                    fake_comment = "Great post!"  # Placeholder (replace with actual comment extraction)
                    c_sentiment, c_confidence = sentiment_analysis(fake_comment)
                    comment_sentiments.append((c_sentiment, c_confidence))
                
                # Aggregate Comment Sentiments
                post["Comments Sentiments"] = comment_sentiments

        return posts
    except Exception as e:
        st.error(f"Error fetching posts: {e}")
        return None

sentiment_pipeline = pipeline("sentiment-analysis")

def sentiment_analysis(data):
    """Classifies the input as pos/neg/neutral."""
    if data:
        result = sentiment_pipeline(data)[0] 
        return result['label'], result['score']
    return "Neutral", 0.0

# streamlit stuff

st.title("Instagram Sentiment Dashboard")

# user input username
username = st.text_input("Enter your public Instagram username: ")


# displaying the data
if username:
    st.subheader(f"Profile Overview: {username}")

    profile_info = get_instagram_profile(username)
    st.write(profile_info)

    st.subheader("📊 Sentiment Analysis of Instagram Posts & Comments")

    posts = fetch_posts(username, max_posts=10)

    if posts:
        df = pd.DataFrame(posts)

        df["Comment Sentiments"] = df["Comments Sentiments"].apply(
            lambda x: ", ".join([f"{s} ({c})" for s, c in x]) if x else "No Comments"
        )

        st.dataframe(df[["Date", "Caption", "Caption Sentiment", "Caption Confidence", "Comment Sentiments"]])

        # Sentiment distribution
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

        # Sentiment trend over time
        st.subheader("📈 Comment Sentiment Trend Over Time")

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        df["Avg Comment Confidence"] = df["Comments Sentiments"].apply(
            lambda x: np.mean([c for _, c in x]) if x else 0
        )

        fig, ax = plt.subplots()
        sns.lineplot(data=df, x="Date", y="Avg Comment Confidence", marker="o", ax=ax)
        plt.xticks(rotation=45)
        plt.title("Average Comment Sentiment Confidence Over Time")
        st.pyplot(fig)