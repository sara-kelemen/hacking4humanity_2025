import streamlit as st
import instaloader
import pandas as pd
import nltk
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
    
def fetch_posts(username:str, max_posts=10):
    """Function to gather posts from the user.

    Args:
        username (str): username
        max_posts (int, optional): the maximum amount of posts to analyze. Defaults to 10.

    Returns:
        posts: all of the posts the function has gathered from the profile
    """
    try:
        posts = get_instagram_posts(username, max_posts)
        return posts
    except Exception as e:
        st.error(f"Error fetching posts: {e}")
        return None
