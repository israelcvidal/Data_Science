import numpy as np
from sklearn import linear_model
import math
from operator import sub

class mlr():
	def predict_value(self, row, coefficients):
		y = coefficients[0]
		for x in range(len([row])):
			y += coefficients[x+1] * [row][x]
		return float(y)

	def estimate_coefficient(self, features, labels, learning_rate, max_iterations):
		coef = [0.0 for i in range(len(features[0]) + 1)]

		for iteration in range(max_iterations):
			total_error = 0
			i=0
			for row in features:
				y = self.predict_value(row, coef)
				error = y-float(labels[i])
				# total_error += error**2
				coef[0] = coef[0] - learning_rate*error
				for j in range(len(row)):
					coef[j+1] = coef[j+1] - learning_rate*error*row[j]
				i+=1
			# print(iteration, learning_rate, total_error)
		return coef

	def fit(self, training_features, training_labels, learning_rate, max_iterations):
		self.learning_rate = learning_rate
		self.max_iterations = max_iterations
		self.labels = training_labels
		self.coefficients = self.estimate_coefficient( training_features, training_labels, learning_rate, max_iterations)

	def predict(self, features):
		predictions = []
		for row in range(len(features)):
			y = self.predict_value(np.array(row), self.coefficients)
			predictions.append(y)
		return predictions

	def total_sum_of_squares(self, labels):
	    mean_labels = np.mean(labels)
	    return sum((y-mean_labels)**2 for y in labels)

	def score(self,features, labels):
	    # return 1.0 - sum((self.labels-self.predict(features))**2)/self.total_sum_of_squares()
	    return 1-sum([(x-y)**2 for x,y in zip(labels, self.predict(features))])/self.total_sum_of_squares(labels)

def main():
	linear_regretion = mlr()

	data = np.loadtxt("aerogerador.txt",delimiter=",")
	rdata = np.random.permutation(data)
	X = rdata[:,0]
	y = rdata[:,1]

	nt = int(len(X) * 0.8)
	X_train = X[:nt].reshape(-1,1)
	X_test = X[nt:].reshape(-1,1)
	y_train = y[:nt]
	y_test = y[nt:]

	regr = linear_model.LinearRegression()
	regr.fit(X_train, y_train)
	print(regr.score(X_test, y_test))

	# dataset = np.array([[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]])

	# features = np.array(dataset[:,0:1])
	# labels = np.array(dataset[:,1:])

	# print(features,labels)
	# X_test = X[nt:].reshape(-1,1)
	# y_train = y[:nt]
	# y_test = y[nt:]

	learning_rate = 0.001
	max_iterations = 100

	linear_regretion.fit(X_train, y_train, learning_rate, max_iterations)
	print(linear_regretion.score(X_train, y_train))
			
main()
