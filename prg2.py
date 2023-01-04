from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('slr.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size=0.1, random_state=6)
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
c= clf.predict(x_test)

acc = accuracy_score(y_test, c)
print("Accuraccy of Decision Tree:",acc)
plot_tree(clf, filled='color')
plt.title("Decision Tree Classifier")
plt.show()