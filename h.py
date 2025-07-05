from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

#simple dataset:[height, weight]
X = [
    [150, 50],
    [160, 60],
    [170, 65],
    [180, 80]
]
#labels : 0 = short, 1=tall
y = [0, 0, 1, 1]

#create and train model
model = DecisionTreeClassifier()
model.fit(X, y)

#predict for same data(for demonstration)
y_pred = model.predict(X)

#compute confusion matrix
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(cm)