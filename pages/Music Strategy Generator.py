import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

st.title("Best Music Strategy")

# Sample extracted data
data = {
    "Artist": [
        "Tyler The Creator", "Anderson .Paak", "Lizzo", "Nipsey Hussle", "Megan Thee Stallion",
        "Nas", "Thundercat", "Beyoncé", "Lil Nas X", "Jay-Z", "Jon Batiste", "Ladysmith Black Mambazo",
        "Tinariwen", "Angélique Kidjo", "Jazmine Sullivan"
    ],
    "Genre": [
        "Rap", "R&B", "Pop", "Rap", "Rap",
        "Rap", "Funk", "R&B HIP HOP", "Hip hop", "Hip hop", "Jazz", "World Music",
        "World Music", "World Music", "R&B"
    ],
    "Album": [
        "Igor", "Ventura", "Cuz I Love You", "Victory Lap", "Fever",
        "King's Disease", "It Is What It Is", "Everything Is Love", "Montero", "The Blueprint 3",
        "WE ARE", "Ilembe: Honoring Shaka Zulu", "Tassili", "Djin Djin", "Heaux Tales"
    ],
    "Won Grammy": [
        1, 1, 1, 0, 0,
        1, 0, 1, 0, 0,
        1, 1, 1, 1, 1
    ],
    "Marketing Strategies": [
        "Cryptic messages, Short videos, Limited-edition merchandise, Collaborations, Media interviews",
        "Collaboration with Bruno Mars, Social media engagement, Live TV shows, Interviews, Merchandise",
        "Re-releasing singles, Collaborations, Behind-the-scenes videos, Viral challenges, High-quality music videos",
        "Own record label, Grassroot engagement, Philanthropy, Collaborations, Documentaries",
        "Remixes, Viral challenges, Striking music videos, Brand endorsements, Good public image",
        "Collaborations, Modern sounds, Engaged fan base, High-quality music, Documentaries",
        "New style, Genre blending, Collaborations, Storytelling videos, Live streams",
        "Surprise release, Music videos, Collaborations, Social campaigns, Exclusive streaming",
        "Tiktok challenge, Controversial statements, Personal themes, Brand partnership, Instagram engagement",
        "Support Black-owned businesses, Social media campaigns, Talk shows, Music tours, Collaborations",
        "Launched on Instagram, Autobiography, Top show collaborations, Apple Music",
        "International partnerships, Touring, Cultural representation, Local languages, Interviews",
        "Touring, Cultural representation, Local languages, Interviews, Documentaries",
        "Cultural representation, Local languages, Social advocacy, Artist collaborations, Workshops",
        "Teasers, Popular themes, Interludes, Community issues, Aesthetic music videos"
    ]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Convert marketing strategies into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Genre'])

# Target variable
y = df['Marketing Strategies']

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X, y)

# Text input for user message
user_message = st.text_input("Enter a music genre:")

# Function to predict marketing strategies based on user input
def predict_marketing_strategy(input_message):
    input_transformed = vectorizer.transform([input_message])
    predicted_strategies = lr_model.predict(input_transformed)
    return predicted_strategies[0].split(', ')  # Split strategies into separate items

if user_message:
    st.subheader("Predicted marketing strategies:")
    predicted_strategies = predict_marketing_strategy(user_message)
    for strategy in predicted_strategies:
        st.markdown(f"- {strategy.strip()}")  # Display each strategy as a bullet point

# Add a button at the bottom
if st.button("Contact us to build out the strategy"):
    st.markdown("[Click here to email us](mailto:Oluwatobiloba.odunuga@stu.cu.edu.ng)")
