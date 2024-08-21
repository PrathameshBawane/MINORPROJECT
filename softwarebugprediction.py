import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# Load the datasets
fix = pd.read_csv('C:/Users/Asus/Downloads/fix.csv', encoding='ISO-8859-1')
fix_train = pd.read_csv('C:/Users/Asus/Downloads/fix_train.csv', encoding='ISO-8859-1')
fix_test = pd.read_csv('C:/Users/Asus/Downloads/fix_test.csv', encoding='ISO-8859-1')
sev = pd.read_csv('C:/Users/Asus/Downloads/sev.csv', encoding='ISO-8859-1')
sev_train = pd.read_csv('C:/Users/Asus/Downloads/sev_train.csv', encoding='ISO-8859-1')
sev_test = pd.read_csv('C:/Users/Asus/Downloads/sev_test.csv', encoding='ISO-8859-1')
embedding = np.load('C:/Users/Asus/Downloads/embedding.npy')
vocab = pd.read_csv('C:/Users/Asus/Downloads/vocab.lst', header=None, names=['word'], encoding='ISO-8859-1')

# Display the first few rows of each dataset
print("fix dataset:")
print(fix.head())

print("\nfix_train dataset:")
print(fix_train.head())

print("\nfix_test dataset:")
print(fix_test.head())

print("\nsev dataset:")
print(sev.head())

print("\nsev_train dataset:")
print(sev_train.head())

print("\nsev_test dataset:")
print(sev_test.head())


# Load and preprocess the corpus(fixsev).txt content
with open('C:/Users/Asus/Downloads/corpus (fixsev).txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

# Basic text preprocessing
corpus = re.sub(r'\s+', ' ', corpus)  # Replace multiple whitespace with single space
corpus = re.sub(r'\d+', ' ', corpus)  # Remove digits

# Tokenize the corpus
vectorizer = CountVectorizer(max_features=1000)
corpus_tokens = vectorizer.fit_transform([corpus]).toarray()

print("Corpus tokens shape:", corpus_tokens.shape)




# Encode the 'Severity' feature in sev datasets
le = LabelEncoder()
sev['Severity_encoded'] = le.fit_transform(sev['Severity'])
sev_train['Severity_encoded'] = le.fit_transform(sev_train['Severity'])
sev_test['Severity_encoded'] = le.fit_transform(sev_test['Severity'])

# Merge sev and fix datasets based on 'Description'
merged_data = sev.merge(fix, on='Description', how='left')
merged_data_train = sev_train.merge(fix_train, on='Description', how='left')
merged_data_test = sev_test.merge(fix_test, on='Description', how='left')

print(merged_data.columns)


# Use embeddings for the 'Description' feature
X = embedding
y = merged_data['Label_y'].values

print(merged_data.head())

print(X.shape)
print(y.shape)

# Ensure that X and y come from the same merged DataFrame
X = merged_data.drop(columns=['Label_x', 'Label_y'])  # Adjust depending on your needs
y = merged_data['Label_y'].values  # Or 'Label_y' based on your previous step

print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Assuming 'embedding' is already loaded and has the necessary features for 'sev_test'
# Prepare the sev_test dataset
X_sev_test = embedding[:sev_test.shape[0]]  # Assuming the same order of rows in embeddings
y_sev_test = sev_test['Label'].values  # Replace 'Label' with the actual label column name in 'sev_test'

# Include the tokenized corpus in the training and test data
# Ensure that corpus_tokens have the same number of rows as X_train and X_test
corpus_tokens_train = corpus_tokens[:X_train.shape[0], :]
corpus_tokens_test = corpus_tokens[X_train.shape[0]:, :]



# Before concatenation, let's diagnose the shapes of the arrays
print("X_test shape:", X_test.shape)
print("corpus_tokens_test shape:", corpus_tokens_test.shape)

# Check if corpus_tokens_test is empty
if corpus_tokens_test.shape[0] == 0:
    print("Error: corpus_tokens_test is empty.")
else:
    # If corpus_tokens_test is not empty, replicate its rows to match X_test if needed
    if corpus_tokens_test.shape[0] == 1:
        corpus_tokens_test = np.repeat(corpus_tokens_test, X_test.shape[0], axis=0)

    # Concatenate only if the dimensions match
    if corpus_tokens_test.shape[0] == X_test.shape[0]:
        X_test = np.concatenate((X_test, corpus_tokens_test), axis=1)
    else:
        print(f"Error: Mismatch in rows. X_test has {X_test.shape[0]} rows, while corpus_tokens_test has {corpus_tokens_test.shape[0]} rows.")

# Final shape check
print("X_test shape after concatenation:", X_test.shape)




# Identify categorical columns that need encoding
categorical_columns = ['Severity']  # Add any other categorical columns here

# One-Hot Encode categorical columns
encoder = OneHotEncoder(sparse_output=False)
encoded_categorical_data = encoder.fit_transform(merged_data[categorical_columns])

# Convert text data to numerical data using TF-IDF Vectorizer
text_data = merged_data['Description']
vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features based on your dataset
X_tfidf = vectorizer.fit_transform(text_data).toarray()

# Drop original categorical columns and text columns from the numerical data
numerical_features = merged_data.drop(columns=categorical_columns + ['Description'])

# Combine TF-IDF features, one-hot encoded categorical data, and numerical features
X_combined = np.hstack((X_tfidf, encoded_categorical_data, numerical_features.values))

# Split the combined data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Ensure that X_train and X_test have the same number of rows as y_train and y_test
assert X_train.shape[0] == y_train.shape[0], f"X_train and y_train row mismatch: {X_train.shape[0]} != {y_train.shape[0]}"
assert X_test.shape[0] == y_test.shape[0], f"X_test and y_test row mismatch: {X_test.shape[0]} != {y_test.shape[0]}"

# Model initialization
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Support Vector Machine": SVC(random_state=42),
    "Naive Bayes": MultinomialNB()
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining and evaluating: {model_name}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot the confusion matrix for each model
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name} Confusion Matrix (Accuracy: {accuracy:.4f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()