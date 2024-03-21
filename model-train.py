import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib

# Load CSV files
combined_df = pd.read_csv('dataset-finalv2.csv')

# Separate features (X) and labels (y)
X = combined_df.iloc[:, :15]  # Assuming the first 15 columns are features
y = combined_df['category']  # Replace 'category_column' with the actual column name for categories

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm_classifier = svm.SVC(kernel='linear')

# Train the model
svm_classifier.fit(X_train, y_train)

# Evaluate the model
accuracy = svm_classifier.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# Export the trained model
joblib.dump(svm_classifier, 'svm_modelv2.pkl')
