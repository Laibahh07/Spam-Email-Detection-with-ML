# Spam-Email-Detection-with-ML
📧 A machine learning project for spam email detection using Python.  Includes text preprocessing, feature engineering, TF-IDF, Logistic Regression,  visualizations, and model saving for real-time email classification.


📧 Spam Email Detection with Machine Learning

A machine learning project to detect Spam vs Ham (legitimate) emails using Python.
This project demonstrates text preprocessing, feature engineering, TF-IDF vectorization, Logistic Regression, and visualizations to analyze email datasets and build a spam classifier.

🚀 Features

Text Preprocessing: Lowercasing, removing URLs, emails, and special characters.

Feature Engineering: Extracts custom features like word count, char count, digit count, stopwords, etc.

TF-IDF Vectorization: Captures word importance for classification.

Model Training: Logistic Regression for efficient spam detection.

Performance Evaluation: Classification report + accuracy score.

Visualizations:

Spam vs Ham distribution

Word count distribution

Average word length by class

Confusion matrix heatmap

Model Saving: Trained model saved as spam_email_detector.joblib for reuse.

📂 Project Structure
spam-email-detection-ml/
│
├── spam.csv                   # Dataset (SMS Spam Collection)
├── spam_detection.ipynb       # Colab/Notebook with full code
├── spam_email_detector.joblib # Saved model
├── README.md                  # Project documentation
└── requirements.txt           # Dependencies


📊 Usage

Run the notebook to train and test the model:

jupyter notebook spam_detection.ipynb


Or use Python script to test the saved model:

import joblib

# Load model
model, tfidf, scaler = joblib.load("spam_email_detector.joblib")

# Example email
email = "Congratulations! You’ve won a $1000 gift card. Click here to claim now."
cleaned = "congratulations you’ve won a gift card click here to claim now"
X_input = tfidf.transform([cleaned])
# Add feature extraction if using combined features

prediction = model.predict(X_input)[0]
print("Spam" if prediction == 1 else "Ham")

📈 Results

Accuracy: ~97–98%

Ham Precision: ~98%

Spam Precision: ~97%

Confusion matrix and performance reports included in notebook.

📷 Sample Outputs

Ham vs Spam distribution

Word count histogram

Confusion matrix heatmap

Classification report


📜 License

This project is licensed under the MIT License.
