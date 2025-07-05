import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#generate random data (same as before)
np.random.seed(0)
student_ids = np.arange(1, 21)
math_scores = np.random.randint(50, 100, size=20)
english_scores = np.random.randint(50, 100, size=20)
science_scores = np.random.randint(50, 100, size=20)
df = pd.DataFrame({
    'StudentID': student_ids,
    'Math': math_scores,
    'English': english_scores,
    'Science': science_scores
})
#features and target
X=df[['English', 'Science']]
y=df['Math']



#split into train and sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

#predict on test set
y_pred = model.predict(X_test)

#evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

#show prediction vs actual
print("\nPredicted Math scores:", y_pred)
print("Actual Math scores:", y_test.values)