#import a dataset
from sklearn.datasets import load_iris
iris = load_iris()


x = iris.data
y = iris.target

#partitioning the data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

#classifier creation number 1
#from sklearn import tree
#my_classifier = tree.DecisionTreeClassifier()

#classifier creation number 2
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier  


my_classifier.fit(x_train, y_train)

#using predictions
predictions = my_classifier.predict(x_test)


#measuring accuracy
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)