#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk


# In[2]:


df = pd.read_csv('Clothing E-Commerce Reviews.csv')


# In[3]:


# Check the first few rows
print(df.head())


# In[4]:


# Define a function for text cleaning
def clean_text(text):
    # Remove non-alphabetic characters and make lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # Tokenize text
    words = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)


# In[6]:


# Function to clean review text
def clean_text(text):
    # Ensure the text is a string, then apply cleaning
    text = str(text)  # Convert to string to avoid errors for numeric values
    text = text.lower()  # Convert to lowercase
    text = ''.join([char if char.isalnum() or char.isspace() else ' ' for char in text])  # Keep alphanumeric and spaces
    return text


# In[8]:


# Apply text cleaning function
df['cleaned_review'] = df['Review Text'].apply(clean_text)


# In[9]:


# Splitting dataset into features and labels
X = df['cleaned_review']  # Features
y = df['Recommended IND']  # Labels



# In[10]:


# Splitting data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[11]:


# Vectorize text data using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# In[12]:


# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)


# In[13]:


# Make predictions
y_pred = model.predict(X_test_tfidf)


# In[14]:


print(df.columns)


# In[15]:


# Make predictions
y_pred = model.predict(X_test_tfidf)


# In[16]:


# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Classification Report:')
print(classification_report(y_test, y_pred))


# In[18]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB  # Import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt


# In[19]:


from sklearn.metrics import f1_score, accuracy_score

# Define a function to evaluate multiple models
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC()
    }
    
    model_performance = {'Model': [], 'Accuracy': [], 'F1-Score': []}

    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')  # Ensure binary F1-score calculation
        
        # Store results
        model_performance['Model'].append(model_name)
        model_performance['Accuracy'].append(accuracy)
        model_performance['F1-Score'].append(f1)
    
    return pd.DataFrame(model_performance)

# Evaluate all models
model_results = evaluate_models(X_train_tfidf, X_test_tfidf, y_train, y_test)

# Print results
print(model_results)

# Visualize performance
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy plot
ax[0].bar(model_results['Model'], model_results['Accuracy'], color='skyblue')
ax[0].set_title('Model Accuracy')
ax[0].set_xlabel('Model')
ax[0].set_ylabel('Accuracy')

# F1-Score plot
ax[1].bar(model_results['Model'], model_results['F1-Score'], color='salmon')
ax[1].set_title('Model F1-Score')
ax[1].set_xlabel('Model')
ax[1].set_ylabel('F1-Score')

plt.tight_layout()
plt.show()


# In[23]:


# Define the models outside the evaluate function for later use
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

# Select the best model based on F1-Score (or you can change to Accuracy)
best_model_name = model_results.loc[model_results['F1-Score'].idxmax()]['Model']
best_model_f1_score = model_results['F1-Score'].max()

# You can also select based on Accuracy instead
# best_model_name = model_results.loc[model_results['Accuracy'].idxmax()]['Model']
# best_model_accuracy = model_results['Accuracy'].max()

print(f"Best Model: {best_model_name} with F1-Score: {best_model_f1_score}")

# Now, train the best model and make predictions
best_model = models[best_model_name]
best_model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred_best = best_model.predict(X_test_tfidf)

# Evaluate the best model again if needed
accuracy_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best, average='binary')

print(f"Best Model's Final Accuracy: {accuracy_best}")
print(f"Best Model's Final F1-Score: {f1_best}")


# In[24]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_best)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix for {best_model_name}')
plt.show()


# In[25]:


from sklearn.metrics import classification_report

# Generate the classification report
report = classification_report(y_test, y_pred_best, target_names=['Negative', 'Positive'])
print(f"Classification Report for {best_model_name}:\n")
print(report)


# In[27]:


# Visualize model performance comparison
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy plot
ax[0].bar(model_results['Model'], model_results['Accuracy'], color='skyblue')
ax[0].set_title('Model Accuracy Comparison')
ax[0].set_xlabel('Model')
ax[0].set_ylabel('Accuracy')
ax[0].set_ylim(0, 1)

# F1-Score plot
ax[1].bar(model_results['Model'], model_results['F1-Score'], color='salmon')
ax[1].set_title('Model F1-Score Comparison')
ax[1].set_xlabel('Model')
ax[1].set_ylabel('F1-Score')
ax[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()


# In[28]:


# Analyze misclassified examples
misclassified = X_test[(y_test != y_pred_best)]
true_labels = y_test[(y_test != y_pred_best)]
predicted_labels = y_pred_best[(y_test != y_pred_best)]

# Display some misclassified examples
print("Some Misclassified Examples:\n")
for i in range(5):  # Display 5 examples
    print(f"Text: {misclassified.iloc[i]}")
    print(f"True Label: {true_labels.iloc[i]}, Predicted Label: {predicted_labels[i]}\n")


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from wordcloud import WordCloud
import nltk
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer    


# In[43]:


import joblib


# In[44]:


# Save the best model (Logistic Regression as an example)
best_model = LogisticRegression()
best_model.fit(X_train_tfidf, y_train)
joblib.dump(best_model, 'sentiment_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')



# In[46]:


# Function to predict sentiment on custom input
def predict_sentiment(review):
    """Predict the sentiment of a custom review input."""
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    cleaned_review = clean_text(review)
    review_tfidf = vectorizer.transform([cleaned_review])
    prediction = model.predict(review_tfidf)
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return sentiment


# # PREDICTION

# In[72]:


# Example usage:
custom_review = "good "
sentiment = predict_sentiment(custom_review)
print(f"The sentiment of the review is: {sentiment}")


# In[67]:


def predict_sentiment(review):
    """Predict the sentiment of a custom review input."""
    # Load the saved model and vectorizer
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    # Clean the review text
    cleaned_review = clean_text(review)
    
    # Debug: Print cleaned review
    print(f"Cleaned Review: {cleaned_review}")
    
    # Transform the cleaned review into TF-IDF representation
    review_tfidf = vectorizer.transform([cleaned_review])
    
    # Debug: Print the TF-IDF representation of the review
    print(f"TF-IDF Representation: {review_tfidf.toarray()}")
    
    # Predict sentiment
    prediction = model.predict(review_tfidf)
    
    # Debug: Print the model prediction
    print(f"Prediction (0=Negative, 1=Positive): {prediction[0]}")
    
    # Convert prediction to readable sentiment
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return sentiment

# Test with a custom review
custom_review = " this shirt is very flattering to all due to the adjustable front tie   "
sentiment = predict_sentiment(custom_review)
print(f"The sentiment of the review is: {sentiment}")


# In[49]:


import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from collections import Counter
import numpy as np

# Check basic info about the dataset
print(df.info())
print(df.head())



# In[50]:


# 1. Sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Recommended IND', data=df, palette='coolwarm')
plt.title('Sentiment Distribution (Positive vs Negative)', fontsize=14)
plt.xlabel('Sentiment (0=Negative, 1=Positive)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()



# In[51]:


# 2. Rating distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Rating', data=df, palette='viridis')
plt.title('Distribution of Ratings', fontsize=14)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()



# Handling Class Imbalance in Sentiment Analysis
# When working with imbalanced sentiment datasets (e.g., more positive than negative comments), the model may become biased toward predicting the majority class. To address this, you can use the following techniques:
# 
# Resampling:
# 
# Oversampling: Increase the number of negative comments using methods like SMOTE.
# Undersampling: Reduce the number of positive comments to balance the classes.
# Class Weight Adjustment:
# 
# Assign higher weights to the minority class (negative comments) to ensure the model pays more attention to them during training.
# Cost-sensitive Learning:
# 
# Assign different misclassification costs to the positive and negative classes, prioritizing correct predictions for the minority class.
# Ensemble Methods:
# 
# Use ensemble models like XGBoost or Random Forest to improve performance, as they can handle class imbalances more effectively.
# Evaluation Metrics:
# 
# Focus on precision, recall, and F1-score rather than accuracy, to better understand how well the model classifies the minority class.
# Threshold Adjustment:
# 
# Adjust the decision threshold to make the model more sensitive to predicting negative comments.
# By applying these strategies, you can ensure a more balanced and fair sentiment analysis, leading to improved model performance for both positive and negative comments.

# In[52]:


# 3. Review Length Distribution
df['review_length'] = df['cleaned_review'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 6))
sns.histplot(df['review_length'], bins=30, kde=True, color='orange')
plt.title('Distribution of Review Lengths', fontsize=14)
plt.xlabel('Review Length (Number of Words)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()



# Insights from Comment Length:
# 
# 
# Sentiment and Length: Longer comments often contain more nuanced feedback, possibly reflecting a mix of positive and negative sentiments. Shorter comments may indicate more extreme sentiments (either positive or negative).
# 
# 
# Engagement: Longer reviews may suggest more engaged or thoughtful users, while shorter reviews could reflect quick judgments.
# 
# 
# Product Feedback: Longer comments are more likely to provide detailed product insights, while shorter ones may focus on overall satisfaction or dissatisfaction.
# 
# 
# Rating Correlation: Review length might correlate with ratings, with shorter reviews potentially reflecting high ratings and longer reviews linked to lower ratings or mixed feedback.
# 
# 
# Actionable Insights: Analyzing the length can help identify patterns in customer sentiment and product feedback, guiding improvements and more accurate sentiment predictions.

# In[53]:


# 4. Most Frequent Words in Positive and Negative Reviews
# Separate the data by sentiment
positive_reviews = df[df['Recommended IND'] == 1]['cleaned_review']
negative_reviews = df[df['Recommended IND'] == 0]['cleaned_review']



# In[54]:


# Combine all reviews in each category
positive_text = ' '.join(positive_reviews)
negative_text = ' '.join(negative_reviews)


# In[55]:


# Create word clouds
wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_text)


# In[56]:


# Display word clouds
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.title('Word Cloud for Positive Reviews', fontsize=14)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.title('Word Cloud for Negative Reviews', fontsize=14)
plt.axis('off')

plt.tight_layout()
plt.show()


# In[58]:


# 5. Most Common Words (with stopwords removed)
def get_most_common_words(text_data):
    """Get the most common words from text data."""
    all_words = ' '.join(text_data).split()
    word_counts = Counter(all_words)
    return word_counts.most_common(10)


# In[59]:


# Most common words in positive reviews
print("Most common words in positive reviews:")
print(get_most_common_words(positive_reviews))


# In[60]:


# Most common words in negative reviews
print("Most common words in negative reviews:")
print(get_most_common_words(negative_reviews))


# In[61]:


negative_text


# In[62]:


positive_text


# In[63]:


df[df['Recommended IND'] == 1]['cleaned_review']


# In[64]:


df[df['Recommended IND'] == 0]['cleaned_review']


# Imbalance: If the dataset is highly imbalanced (e.g., too many positive reviews), the model might be predicting the majority class most of the time. Fixes:
# 
# Resampling techniques like SMOTE (for oversampling the minority class) or undersampling the majority class.
# Use class weights in your model training (e.g., class_weight='balanced' in LogisticRegression).
# Review Length: If reviews are too short or too long, consider truncating or padding reviews to ensure consistency across all data points.
# 
# Key Words in Reviews: If specific words appear predominantly in one sentiment, you might want to tweak your model's feature extraction or try more sophisticated models that can capture context (e.g., BERT, LSTM).
# 
# Model Performance on Imbalanced Data: If the model still struggles despite addressing imbalances, try different algorithms that are more robust to imbalance, or adjust the decision threshold for classification.

# In[ ]:





# In[ ]:





# In[ ]:




