# Choose a social media platform and enter the account name for a personalized dashboard (must be public)
# Prototype: instagram
import instaloader

# Initialize Instaloader
L = instaloader.Instaloader()

# Fetch profile details
username = input("Enter username: ")
profile = instaloader.Profile.from_username(L.context, username)

print(f"Username: {profile.username}")
print(f"Full Name: {profile.full_name}")
print(f"Followers: {profile.followers}")
print(f"Following: {profile.followees}")
print(f"Bio: {profile.biography}")

for post in profile.get_posts():
    print(f"Post Date: {post.date_utc}")
    print(f"Caption: {post.caption}")
    print(f"Likes: {post.likes}")
    print(f"Comments: {post.comments}")
    print(f"Post URL: {post.url}")
    print("-" * 40)
# this worked on my account for testing