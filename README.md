# SENTIMENT-ANALYSIS-WITH-NLP
**COMPANY**        : CODTECH IT SOLUTIONS
**NAME**           : EDAMALAPATI MOTHILAL CHOWDARY
**INTERN ID**      : CT12JJP
**DOMAIN**         : MACHINE LEARNING
**BATCH DURATION** : January 5th,2025 to March 5th,2025
**MENTOR NAME**    : NEELA SANTHOSH

# DESCRIPTION OF TASK :
**Sentiment Analysis on IMDB Dataset Using Logistic Regression**
    This project focuses on performing sentiment analysis on the IMDB dataset, which contains movie reviews labeled as either "positive" or "negative." The goal is to develop a machine learning model that can classify new reviews based on their sentiment. The method used for classification is Logistic Regression, a popular algorithm for binary classification tasks.

**1. Loading and Understanding the Dataset**
The dataset, obtained from Kaggle, includes two main columns:

Review: The actual text of the movie review.
Sentiment: The label indicating whether the review is positive or negative.
The dataset is first loaded and inspected for structure, checking for potential issues like missing values or inconsistent formatting.

**2. Text Preprocessing**
Before using text data for machine learning, it needs to be cleaned and structured. The preprocessing steps include:

Lowercasing: Converting all text to lowercase to avoid duplication of words like “Great” and “great.”
Removing Special Characters: Eliminating punctuation and symbols that do not affect sentiment.
Removing Extra Spaces: Ensuring uniform formatting by cleaning unnecessary whitespace.
Stopword Removal: Removing common but uninformative words (e.g., “the,” “and,” “is”).
The cleaned text is stored in a new column, allowing easy comparison with the original data.

**3. Splitting the Data**
To evaluate the model’s performance, the data is divided into a training set (80%) and a testing set (20%). The training set is used to build the model, while the testing set is used to assess how well the model generalizes to new, unseen data.

**4. Feature Extraction with TF-IDF**
Machine learning models require numerical inputs, so the text data is converted into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency). TF-IDF assigns a weight to each word, based on how often it appears in a document and how rare it is across all documents. This helps the model focus on meaningful words while downplaying common ones that provide little information about sentiment.

**5. Training the Logistic Regression Model**
With the transformed data, a Logistic Regression model is trained to classify reviews as either positive or negative. Logistic Regression is a straightforward and effective model for binary classification problems. The model learns from the training data and forms a relationship between the features (words) and the sentiment labels.

**6. Evaluating Model Performance**
After training, the model’s performance is evaluated using the test set. Several metrics are used to assess the model:

Accuracy: The percentage of correct predictions.
Precision: The proportion of positive predictions that are actually positive.
Recall: The proportion of actual positives that are correctly identified.
F1-score: A balanced metric between precision and recall, useful for imbalanced datasets.
A confusion matrix is also generated, helping visualize how well the model detects positive and negative reviews.

**7. Predicting Sentiment for New Reviews**
Once trained and evaluated, the model is tested on new, unseen reviews. These reviews undergo the same preprocessing steps, then the model predicts whether each review is positive or negative. For example:

“Absolutely love it! Best purchase ever.” → Positive
“Horrible, I regret buying this.” → Negative
This demonstrates how the model can automatically classify real-world reviews.

**Key Insights and Findings**
**Preprocessing Impact:** Text cleaning and stopword removal improve model performance by eliminating irrelevant words.
**TF-IDF:** TF-IDF effectively captures important words for sentiment classification while filtering out common, uninformative ones.
**Logistic Regression:** A simple yet effective model that provides strong results for binary classification.
**Model Evaluation:** Multiple metrics like accuracy, precision, recall, and confusion matrix provide a more comprehensive evaluation of the model's performance.
**Practical Applications:** The model can be used in various domains like customer feedback analysis, social media monitoring, and content moderation.
**Conclusion**
    This sentiment analysis project successfully classifies IMDB movie reviews using Logistic Regression. By following a structured process—data preprocessing, feature extraction, model training, and evaluation—the model achieves high accuracy and can generalize well to new reviews. 
