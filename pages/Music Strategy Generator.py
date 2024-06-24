import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

st.title("Model Analysis")

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
X = vectorizer.fit_transform(df['Marketing Strategies'])

# Target variable
y = df['Won Grammy']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

# Evaluation metrics
metrics = ['accuracy', 'precision', 'recall', 'f1']

# Evaluate models using cross-validation
results = {}
for model_name, model in models.items():
    model_results = {}
    for metric in metrics:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=metric)
        model_results[metric] = np.mean(scores)
    results[model_name] = model_results

# Calculate the average score for each model
average_scores = {model_name: np.mean(list(metrics.values())) for model_name, metrics in results.items()}

# Find the best model based on the average score of all metrics
best_model_name = max(average_scores, key=average_scores.get)
best_model_score = average_scores[best_model_name]

# Prepare data for display
results_data = {
    "Model": list(results.keys()),
    "Accuracy": [metrics['accuracy'] for metrics in results.values()],
    "Precision": [metrics['precision'] for metrics in results.values()],
    "Recall": [metrics['recall'] for metrics in results.values()],
    "F1 Score": [metrics['f1'] for metrics in results.values()]
}
results_df = pd.DataFrame(results_data)

average_scores_data = {
    "Model": list(average_scores.keys()),
    "Average Score": list(average_scores.values())
}
average_scores_df = pd.DataFrame(average_scores_data)

# Displaying the results as a table
st.subheader("Model Performance")
st.dataframe(results_df.set_index('Model').style.set_properties(**{'text-align': 'center'}).set_table_styles([{
    'selector': 'th',
    'props': [('text-align', 'center')]
}]))

# Displaying the average scores as a table
st.subheader("Average Scores")
st.dataframe(average_scores_df.set_index('Model').style.set_properties(**{'text-align': 'center'}).set_table_styles([{
    'selector': 'th',
    'props': [('text-align', 'center')]
}]))

# Plotting the results
st.subheader("Model Performance Visualization")
fig, ax = plt.subplots()
results_df.set_index('Model').plot(kind='bar', y=['Accuracy', 'Precision', 'Recall', 'F1 Score'], ax=ax)
ax.set_ylabel('Scores')
plt.xticks(rotation=45)
st.pyplot(fig)

# Display the best model
st.subheader(f"Best model based on overall average score: {best_model_name}")

# Add a button at the bottom
if st.button("Contact us to build out the strategy"):
    st.write("Thank you for your interest! Please contact us at [your-email@example.com](mailto:your-email@example.com) to build out the strategy.")
# Add a button at the bottom
if st.button("Contact us to build out the strategy"):
    st.markdown("[Click here to email us](mailto:Oluwatobiloba.odunuga@stu.cu.edu.ng)")
