import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render
import numpy as np
from keras.models import load_model
from instaloader import Instaloader, Profile, ProfileNotExistsException
import pandas as pd
import os
import csv
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# File paths and credentials
dataset_file = 'train.csv'
model_file = 'model.pkl'
instagram_data_file = 'instagram_data.csv'
insta_username = 'youdoyou_123456'
insta_password = 'icandoit'
session_file = ".instaloader-session-youdoyou_123456"
celebrity_usernames = ['salmankhan', 'shahrukhkhan', 'deepikapadukone']
# Path to the session file
session_file = os.path.abspath(session_file)

# Verifying session file existence
print(os.path.exists(session_file))  # Should return True if the session file exists
print(os.getcwd())  # Check current working directory

def login_to_instagram():
    """Logs into Instagram using Instaloader session or manual login."""
    L = Instaloader(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36')

    try:
        if os.path.exists(session_file):
            L.load_session_from_file(insta_username, filename=session_file)
            print("Successfully logged in using session.")
        else:
            print("Session file not found. Logging in manually.")
            L.context.login(insta_username, insta_password)
            L.save_session_to_file(filename=session_file)
            print("Login successful, session saved.")
        return L
    except Exception as e:
        print(f"Error during login: {str(e)}")
        return None

def Index(request):
    """Renders the index page."""
    return render(request, "fpd/instagram.html")

def insta(request):
    """Handles Instagram login and renders the Instagram data page."""
    login_to_instagram()
    return render(request, 'fpd/instagram.html')

def preprocess_data(profile):
    """Prepares the profile data for model prediction."""
    # Extracting relevant features from the profile
    username = profile.username
    profile_pic = 1 if profile.profile_pic_url else 0
    nums_username_length = len(username) / 30
    fullname_words = len(profile.full_name.split()) if profile.full_name else 0
    nums_length_fullname = len(profile.full_name) / 30 if profile.full_name else 0
    name_equals_username = int(profile.full_name.replace(" ", "").lower() == username.lower()) if profile.full_name else 0
    description_length = len(profile.biography) if profile.biography else 0
    external_URL = 1 if profile.external_url else 0
    is_private = int(profile.is_private)
    num_posts = profile.mediacount
    num_followers = profile.followers
    num_followees = profile.followees
    blue_tick = 1 if profile.is_verified else 0
    follower_following_ratio = num_followers / num_followees if num_followees > 0 else 0
    post_per_follower = num_posts / num_followers if num_followers > 0 else 0

    # Compile all extracted features into a list
    features = [
        profile_pic, 
        nums_username_length, 
        fullname_words, 
        nums_length_fullname, 
        name_equals_username, 
        description_length, 
        external_URL, 
        is_private, 
        num_posts, 
        num_followers, 
        num_followees, 
        blue_tick, 
        follower_following_ratio, 
        post_per_follower
    ]
    
    print("Extracted Features:", features)
    
    # Return the features as a numpy array for prediction
    return np.array(features)

def save_to_dataset(profile, result, instagram_data_file):
    """Saves profile data to the CSV dataset along with the prediction result as 0 or 1."""
    if not os.path.exists(instagram_data_file):
        with open(instagram_data_file, 'w', newline='') as csvfile:
            fieldnames = ['username', 'mediacount', 'followers', 'followees', 
                          'has_viewable_story', 'language', 'new_feature', 
                          'profile_pic', 'nums_username_length', 'fullname_words',
                          'nums_length_fullname', 'name_equals_username', 'description_length',
                          'external_URL', 'private', '#posts', '#followers', '#follows',
                          'blue_tick', 'follower_following_ratio', 'post_per_follower', 'result']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Get the processed features
    username = profile.username
    mediacount = int(profile.mediacount)
    followers = int(profile.followers)
    followees = int(profile.followees)
    has_story = int(profile.has_viewable_story)
    lang_num = 5  # Arbitrary value
    new_feature = 42  # Placeholder for any additional feature
    
    # Features from preprocess_data
    profile_pic = 1 if profile.profile_pic_url else 0
    nums_username_length = len(username) / 30
    fullname_words = len(profile.full_name.split()) if profile.full_name else 0
    nums_length_fullname = len(profile.full_name) / 30 if profile.full_name else 0
    name_equals_username = int(profile.full_name.replace(" ", "").lower() == username.lower()) if profile.full_name else 0
    description_length = len(profile.biography) if profile.biography else 0
    external_URL = 1 if profile.external_url else 0
    is_private = int(profile.is_private)
    num_posts = profile.mediacount
    num_followers = profile.followers
    num_followees = profile.followees
    blue_tick = 1 if profile.is_verified else 0
    follower_following_ratio = num_followers / num_followees if num_followees > 0 else 0
    post_per_follower = num_posts / num_followers if num_followers > 0 else 0

    # Convert result to 0 or 1
    result_value = 0 if result == "The Profile is Fake" else 1

    # Save to CSV
    with open(instagram_data_file, 'a', newline='') as csvfile:
        fieldnames = ['username', 'mediacount', 'followers', 'followees', 
                      'has_viewable_story', 'language', 'new_feature', 
                      'profile_pic', 'nums_username_length', 'fullname_words',
                      'nums_length_fullname', 'name_equals_username', 'description_length',
                      'external_URL', 'private', '#posts', '#followers', '#follows',
                      'blue_tick', 'follower_following_ratio', 'post_per_follower', 'result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'username': username,
            'mediacount': mediacount,
            'followers': followers,
            'followees': followees,
            'has_viewable_story': has_story,
            'language': lang_num,
            'new_feature': new_feature,
            'profile_pic': profile_pic,
            'nums_username_length': nums_username_length,
            'fullname_words': fullname_words,
            'nums_length_fullname': nums_length_fullname,
            'name_equals_username': name_equals_username,
            'description_length': description_length,
            'external_URL': external_URL,
            'private': is_private,
            '#posts': num_posts,
            '#followers': num_followers,
            '#follows': num_followees,
            'blue_tick': blue_tick,
            'follower_following_ratio': follower_following_ratio,
            'post_per_follower': post_per_follower,
            'result': result_value  # Save the result as 0 or 1
        })

# Load dataset and preprocess
data = pd.read_csv('insta_test.csv')
data.fillna(0, inplace=True)

# Features and target
X = data.drop('fake', axis=1)
y = data['fake']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import pandas as pd
import numpy as np
import pandas as pd
def test_instagram_profile(profile_data, username,profile):
    """Tests the Instagram profile data against predefined criteria and the trained model."""
    
    # Convert ndarray to dictionary if necessary (already done earlier)
    if isinstance(profile_data, np.ndarray):
        profile_data = dict(zip(X.columns, profile_data.flatten()))

    print("Available keys in profile data:", profile_data.keys())

    blue_tick = 1 if profile.is_verified else 0
    # Special check for verified profiles
    followers = profile_data.get('#followers', 0)
    follows = profile_data.get('#follows', 0)
  

    print(f"Followers: {followers}, Follows: {follows}, Blue Tick: {blue_tick}")

    if followers > 1000 and follows < 10 and blue_tick == 1:
        return "Real Profile (Blue Tick, High Followers, Low Following)"
    # Check for fake user profiles with low activity
    if profile_data.get('#followers', 0) < 250 and profile_data.get('#posts', 0) < 100:
        return "Fake User Profile (Low Activity)"
    
    # Check for normal user profiles with low activity
    if profile_data.get('#followers', 0) < 500 and profile_data.get('#posts', 0) < 10:
        return "Normal User Profile (Low Activity)"
    print(profile_data.get('follower_following_ratio', 0))
    
    # Check for profiles with reasonable follower-following ratios
    if profile_data.get('follower_following_ratio', 0) < 3 and profile_data.get('#posts', 0) > 5:
        return "Normal User Profile (Balanced Following and Posts)"
    
    # Check for potential fan pages
    if any(celeb in username.lower() for celeb in celebrity_usernames):
        if "fanpage" in profile_data.get('biography', '').lower():
            return "Potential Fan Page (Fanpage Mentioned in Bio)"
        elif "fanpage" in profile_data.get('full_name', '').lower():
            return "Potential Fan Page (Fanpage Mentioned in Full Name)"
        elif profile_data.get('#followers', 0) < 100:
            return "Potential Fan Page (Low Activity)"
        elif profile_data.get('#followers', 0) >= 100 and profile_data.get('#followers', 0) < 1000:
            return "Potential Fan Page (Moderate Followers)"
        elif profile_data.get('#followers', 0) >= 1000:
            return "Potential Fan Page (High Followers)"
    
    # Convert profile data to DataFrame
    profile_df = pd.DataFrame([profile_data])
    
    # Ensure all required features are present
    required_features = X.columns.tolist()  # X is the training dataset used to fit the model
    profile_df = profile_df.reindex(columns=required_features, fill_value=0)  # Fix column reordering issue
    
    # Predict using the trained model
    prediction = model.predict(profile_df)
    
    # Output the result with detailed reasons if the profile is predicted as fake
    if prediction[0] == 1:  # Fake Profile
        reasons = []
        if profile_data.get('blue_tick', 0) == 0:
            reasons.append("Profile is not verified (no blue tick).")
        if profile_data.get('#followers', 0) < 100:
            reasons.append("Low follower count.")
        if profile_data.get('#posts', 0) < 5:
            reasons.append("Low posting activity.")
        if profile_data.get('follower_following_ratio', 0) > 100:  # Adjust threshold as needed
            reasons.append("Suspicious follower-following ratio.")
        if any(celeb in username.lower() for celeb in celebrity_usernames):
            reasons.append("Username resembles a celebrity, which may indicate impersonation.")
        
        return f"Fake Profile. Reasons: {', '.join(reasons)}"
    else:
        return "Real Profile"




def instagram(request):
    """Handles the Instagram profile detection process."""
    if request.method == 'POST':
        input_username = request.POST.get("username", "").strip()

        if not input_username:
            msg = "Please provide a valid username."
            return render(request, 'fpd/instagram.html', {'msg': msg})

        try:
            # Attempt to login to Instagram session
            L = login_to_instagram()  # Assuming login_to_instagram handles the session
            if not L:
                msg = "Failed to log in to Instagram."
                return render(request, 'fpd/instagram.html', {'msg': msg})

            try:
                # Retrieve the profile from Instagram
                profile = Profile.from_username(L.context, input_username)
            except ProfileNotExistsException:
                msg = "The provided Instagram profile does not exist."
                return render(request, 'fpd/instagram.html', {'msg': msg})

            if profile:
                try:
                    # Preprocess the profile data
                    profile_data = preprocess_data(profile)

                    # Test the Instagram profile
                    result = test_instagram_profile(profile_data, input_username, profile)

                    # Save the result to dataset, passing instagram_data_file
                    save_to_dataset(profile, result, instagram_data_file)

                    msg = f"The profile '{input_username}' is : {result}"

                except Exception as e:
                    msg = f"An error occurred during profile analysis: {str(e)}"
            else:
                msg = "No profile found for the username provided."
        except Exception as e:
            msg = f"An error occurred: {str(e)}"
        
        return render(request, 'fpd/instagram.html', {'msg': msg})