import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import itce
import joblib

# data = pd.read_excel("tickets.xlsx")
# print(data.shape)

df_balanced = itce.another_main()
X = df_balanced["cleaned_txt"]
y = df_balanced["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train the Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train_tfidf, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

#save this model
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(clf, "model.pkl")

