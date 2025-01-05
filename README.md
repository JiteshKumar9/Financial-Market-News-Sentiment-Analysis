# Financial Market News - Sentiment Analysis

This project aims to predict the overall sentiment (positive or negative) of financial news articles. The dataset contains the top 25 news articles for each day, and the task is to train a model to predict the sentiment based on the news content.

## Dataset

The dataset consists of 4101 entries and 27 columns. The columns include:

- **Date**: The date of the news articles.
- **Label**: Sentiment label (0 = Negative, 1 = Positive).
- **News 1 - News 25**: The news articles (top 25 news items for each day).

### Example of the dataset:
| Date       | Label | News 1 | News 2 | News 3 | ... |
|------------|-------|--------|--------|--------|-----|
| 01-01-2010 | 0     | McIlroy's men catch cold from Gudjonsson | Obituary: Brian Walsh | Workplace blues leave employers in the red | ... |
| 02-01-2010 | 0     | Warning from history points to crash | Investors flee to dollar haven | Banks and tobacco in favour | ... |
| 03-01-2010 | 0     | Comment: Why Israel's peaceniks feel betrayed | Court deals blow to seizure of drug assets | An ideal target for spooks | ... |

## Steps for Model Training

### 1. Import Libraries
```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

2. Import Dataset
python
Copy code
df = pd.read_csv(r'https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Financial%20Market%20News.csv', encoding="ISO-8859-1")
3. Data Overview
python
Copy code
df.head()  # Display the first few rows of the dataset
df.info()  # Dataset information (columns, data types, missing values)
df.shape  # Dataset shape (number of rows and columns)
df.columns  # List of column names
4. Feature Selection
Combine the news columns into a single list of news articles for each row:

python
Copy code
news = []
for row in range(0, len(df.index)):
    news.append(' '.join(str(x) for x in df.iloc[row, 2:27]))
5. Text Conversion to Bag of Words
Convert the news articles into a bag-of-words representation:

python
Copy code
cv = CountVectorizer(lowercase=True, ngram_range=(1, 1))
X = cv.fit_transform(news)
6. Train-Test Split
Split the dataset into training and testing sets:

**python code**
X_train, X_test, y_train, y_test = train_test_split(X, df['Label'], test_size=0.3, stratify=df['Label'], random_state=2529)

7. Train Random Forest Classifier
Train a Random Forest Classifier on the training data:

**python CODE**
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)

8. **Model Evaluation**
Make predictions and evaluate the model:

python code
y_pred = rf.predict(X_test)

# Confusion Matrix
confusion_matrix(y_test, y_pred)

# Classification Report
print(classification_report(y_test, y_pred))

9. **Evaluation Metrics**
Confusion Matrix:
[[136  445]
 [170  480]]


## Classification Report:
text
Copy code
              precision    recall  f1-score   support

           0       0.44      0.23      0.31       581
           1       0.52      0.74      0.61       650

    accuracy                           0.50      1231
   macro avg       0.48      0.49      0.46      1231
weighted avg       0.48      0.50      0.47      1231

##Conclusion
The model achieves an accuracy of approximately 50%. The positive class (1) is predicted with higher recall, while the negative class (0) has lower precision and recall. Further model tuning and data preprocessing can be explored to improve the results.

##Requirements
**pandas
numpy
scikit-learn
matplotlib (optional for visualization)**
