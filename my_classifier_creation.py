from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b)

class scrapyknn():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions
    def closest(self, row):
        best_distance = euc(row, self.x_train)
        best_index = 0
        for i in range(1, len(self.x_train)):
            dist = euc(row, self.x_train)
            if dist<best_distance:
                best_distance = dist
                best_index = i
        return self.y_train[best_index]
    



#import a dataset
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

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
#from sklearn.neighbors import KNeighborsClassifier
my_classifier = scrapyknn() 


my_classifier.fit(x_train, y_train)

#using predictions
predictions = my_classifier.predict(x_test)


#measuring accuracy
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)