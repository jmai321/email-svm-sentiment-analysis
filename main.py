import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from textblob import TextBlob

# Reads in the dataset
df = pd.read_csv("spam_ham_dataset.csv")

# Split the dataset into training and testing
# Training Dataset
x = df["text"]
y = df["label"]
x_train, y_train = x[0:1000], y[0:1000]

# Testing Dataset
x_test, y_test = x[1000:], y[1000:]

# Turns email text into vector numbers
cv = CountVectorizer()
features = cv.fit_transform(x_train)

# Model and tuning for the best parameters
tuned_param = {
    'kernel': ['linear', 'rbf'], 
    'gamma': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
    'C': [1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
}
model = GridSearchCV(svm.SVC(), tuned_param, return_train_score=True, cv=3)
model.fit(features, y_train)

# Print best parameters
print("Best Parameters:", model.best_params_)

# Prints Accuracy of Model
features_test = cv.transform(x_test)
print("Accuracy of Spam Dataset:", model.score(features_test, y_test))

results = pd.DataFrame(model.cv_results_)
results = results[['param_C', 'param_gamma', 'mean_test_score']]
results = results.groupby(['param_C', 'param_gamma'], as_index=False).mean()
pivot_table = results.pivot(index='param_C', columns='param_gamma', values='mean_test_score')

plt.figure(figsize=(10, 8))
plt.title("Grid Search Accuracy by Parameters")
plt.xlabel("Gamma")
plt.ylabel("C")
plt.imshow(pivot_table, interpolation='nearest', cmap='viridis')
plt.colorbar(label='Accuracy')
plt.xticks(range(len(pivot_table.columns)), pivot_table.columns, rotation=45)
plt.yticks(range(len(pivot_table.index)), pivot_table.index)
plt.tight_layout()
plt.show()

# Filter spam emails for sentiment analysis
spam_emails = df[df["label"] == "spam"]

# Perform sentiment analysis on spam emails
spam_emails["sentiment"] = spam_emails["text"].apply(lambda text: TextBlob(text).sentiment.polarity)

# Categorize sentiment
def categorize_sentiment(polarity):
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

spam_emails["sentiment_category"] = spam_emails["sentiment"].apply(categorize_sentiment)

# Plot sentiment distribution
plt.figure(figsize=(8, 6))
spam_sentiment_counts = spam_emails["sentiment_category"].value_counts()
plt.bar(spam_sentiment_counts.index, spam_sentiment_counts.values, color=['green', 'blue', 'red'])
plt.title("Sentiment Distribution in Spam Emails")
plt.xlabel("Sentiment Category")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
