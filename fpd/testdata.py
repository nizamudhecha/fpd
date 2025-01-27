import instaloader
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Initialize instaloader
loader = instaloader.Instaloader()

# List of known celebrity usernames or names (You can expand this list)
celebrity_usernames = ['salmankhan', 'shahrukhkhan', 'deepikapadukone']  # Example

# Function to fetch profile data based on username
def fetch_profile_data(username):
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        profile_data = {
            'profile_pic': 1 if profile.profile_pic_url else 0,
            'nums_username_length': len(username) / 30,
            'fullname_words': len(profile.full_name.split()) if profile.full_name else 0,
            'nums_length_fullname': len(profile.full_name) / 30 if profile.full_name else 0,
            'name_equals_username': int(profile.full_name.replace(" ", "").lower() == username.lower()) if profile.full_name else 0,
            'description_length': len(profile.biography) if profile.biography else 0,
            'external_URL': 1 if profile.external_url else 0,
            'private': int(profile.is_private),
            '#posts': profile.mediacount,
            '#followers': profile.followers,
            '#follows': profile.followees,
            'blue_tick': 1 if profile.is_verified else 0,
            'follower_following_ratio': profile.followers / profile.followees if profile.followees > 0 else 0,
            'post_per_follower': profile.mediacount / profile.followers if profile.followers > 0 else 0
        }
        return profile_data
    except Exception as e:
        print(f"Error fetching data for {username}: {e}")
        return None

# Load the dataset
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

# Function to test the Instagram profile data
def test_instagram_profile(profile_data, username):
    # Special check for verified profiles
    if profile_data['#followers'] > 1000 and profile_data['#follows'] < 10 and profile_data['blue_tick'] == 1:
        return "Real Profile (Blue Tick, High Followers, Low Following)"
    
    if profile_data['#followers'] < 250 and profile_data['#posts'] < 100:
        return "Fake User Profile (Low Activity)"
    
    # Check for normal user behavior
    if profile_data['#followers'] < 500 and profile_data['#posts'] < 10:
        return "Normal User Profile (Low Activity)"
    
    # Check for profiles with reasonable follower-following ratios
    if profile_data['follower_following_ratio'] < 3 and profile_data['#posts'] > 5:
        return "Normal User Profile (Balanced Following and Posts)"
    
    
    
    # Check for potential fan pages
    if any(celeb in username.lower() for celeb in celebrity_usernames):
        if "fanpage" in profile_data.get('biography', '').lower():
            return "Potential Fan Page"  # Check for 'fanpag' in bio
        elif "fanpage" in profile_data.get('full_name','').lower():
            return "Potential Fan Page"  # Check for 'fanpag' in full name
        elif profile_data['#followers'] < 100:  # Adjust threshold for fan pages
            return "Potential Fan Page (Low Activity)"
        elif profile_data['#followers'] >= 100 and profile_data['#followers'] < 1000:
            return "Potential Fan Page (Moderate Followers)"
        elif profile_data['#followers'] >= 1000:
            return "Potential Fan Page (High Followers)"
    
    # Convert profile data to DataFrame
    profile_df = pd.DataFrame([profile_data])
    
    # Ensure all required features are present
    required_features = X.columns.tolist()
    for feature in required_features:
        if feature not in profile_df.columns:
            profile_df[feature] = 0  # Assign default value if missing
    
    # Reorder columns to match training data
    profile_df = profile_df[required_features]
    
    # Predict using the trained model
    prediction = model.predict(profile_df)
    
    # Output the result
    if prediction[0] == 1:
        reasons = []
        if profile_data['blue_tick'] == 0:
            reasons.append("Profile is not verified (no blue tick).")
        if profile_data['#followers'] < 100:
            reasons.append("Low follower count.")
        if profile_data['#posts'] < 5:
            reasons.append("Low posting activity.")
        if profile_data['follower_following_ratio'] > 100:  # Adjust threshold as needed
            reasons.append("Suspicious follower-following ratio.")
        if any(celeb in username.lower() for celeb in celebrity_usernames):
            reasons.append("Username resembles a celebrity, which may indicate impersonation.")
        
        return f"Fake Profile. Reasons: {', '.join(reasons)}"
    else:
        return "Real Profile"

# Get username input
username = input("Enter the Instagram username: ")

# Fetch profile data based on username
profile_data = fetch_profile_data(username)

if profile_data:
    # Display fetched profile data (optional)
    print("\nFetched Profile Data:")
    for key, value in profile_data.items():
        print(f"{key}: {value}")
    
    # Test the Instagram profile
    result = test_instagram_profile(profile_data, username)
    print(f"\nThe profile '{username}' is classified as: {result}")
else:
    print("Failed to retrieve profile data.")
