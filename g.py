import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate a larger synthetic dataset
np.random.seed(0)
num_students = 200  # Increased from 20 to 200
student_ids = np.arange(1, num_students + 1)
english_scores = np.random.randint(50, 100, size=num_students)
science_scores = np.random.randint(50, 100, size=num_students)
noise = np.random.normal(0, 5, size=num_students)
math_scores = 0.5 * english_scores + 0.3 * science_scores + 10 + noise
math_scores = math_scores.round().astype(int)
passed = (math_scores >= 75).astype(int)  # 1 = pass, 0 = fail

df = pd.DataFrame({
    'StudentID': student_ids,
    'Math': math_scores,
    'English': english_scores,
    'Science': science_scores,
    'Passed': passed
})

# 2. Features and target
X = df[['English', 'Science']]
y = df['Passed']

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Decision Tree Classifier
print('--- Decision Tree Classifier ---')
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, dt_pred))
print('Classification Report:\n', classification_report(y_test, dt_pred))

# 5. Random Forest Classifier
print('--- Random Forest Classifier ---')
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, rf_pred))
print('Classification Report:\n', classification_report(y_test, rf_pred))