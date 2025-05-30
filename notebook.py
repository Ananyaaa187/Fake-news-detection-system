pip install pandas numpy scikit-learn matplotlib seaborn wordcloud nltk
# Import required libraries
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Download NLTK stopwords
nltk.download('stopwords')

# Step 1: Create Sample Dataset and Save as news.csv
data = {
    'title': [
        "Fake News: Celebrity cloned!",
        "Real News: New healthcare plan passed",
        "Fake: Moon landing was staged",
        "Real: Scientists discover new species",
        "Fake: Earth is flat",
        "Real: Tech company announces new AI chip"
    ],
    'text': [
        "A popular celebrity was cloned using secret alien technology, experts say.",
        "The government has passed a new healthcare reform bill after months of debate.",
        "New evidence suggests the 1969 moon landing was filmed in a studio.",
        "Researchers in the Amazon have discovered a previously unknown species of frog.",
        "Influencers reignite debate claiming Earth is flat despite scientific proof.",
        "A major tech company has announced its latest chip designed for AI processing."
    ],
    'label': ["FAKE", "REAL", "FAKE", "REAL", "FAKE", "REAL"]
}

df = pd.DataFrame(data)
df.to_csv("news.csv", index=False)
print("✅ news.csv created successfully.")

# Step 2: Load the dataset
df = pd.read_csv("news.csv")

# Combine title and text into one field
df["content"] = df["title"] + " " + df["text"]

# Convert labels to binary (FAKE = 0, REAL = 1)
df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})

# Step 3: Clean the text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single characters
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
    text = text.lower()
    return text

df["cleaned"] = df["content"].apply(clean_text)

# Step 4: WordCloud Visualization
text_fake = ' '.join(df[df['label'] == 0]['cleaned'])
text_real = ' '.join(df[df['label'] == 1]['cleaned'])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(WordCloud(stopwords=stopwords.words('english'), background_color='white').generate(text_fake))
plt.title("Fake News WordCloud")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(WordCloud(stopwords=stopwords.words('english'), background_color='white').generate(text_real))
plt.title("Real News WordCloud")
plt.axis('off')

plt.tight_layout()
plt.show()

# Step 5: Vectorize Text
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
X = tfidf.fit_transform(df["cleaned"]).toarray()
y = df["label"]

# Step 6: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 8: Evaluate Model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# Step 9: Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
