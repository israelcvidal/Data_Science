import numpy as np
from sklearn import linear_model
import math
from operator import sub
import tools as t

class mlr():
	def predict_value(self, row, coefficients):
		# print(coefficients, row)
		return coefficients[0] + sum(x*y for x,y in zip(coefficients[1:], row) )
		
	def estimate_coefficient(self, features, labels, learning_rate, max_iterations):
		coef = [0.0 for i in range(len(features[0]) + 1)]
		for iteration in range(max_iterations):
			total_error = 0
			i=0
			for row in features:
				y = self.predict_value(row, coef)
				error = y-float(labels[i])
				total_error += error**2
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
		return self.coefficients

	def predict(self, features):
		predictions = []
		for row in features:
			y = self.predict_value(row, self.coefficients)
			predictions.append(y)
		return predictions

	def total_sum_of_squares(self, labels):
	    mean_labels = np.mean(labels)
	    return sum((y-mean_labels)**2 for y in labels)

	def score(self,features, labels):
	    # return 1.0 - sum((self.labels-self.predict(features))**2)/self.total_sum_of_squares()
	    return float(1-sum([(float(x)-float(y))**2 for x,y in zip(labels, self.predict(features))])/self.total_sum_of_squares(labels))

  	def normalize():
  		pass

def main():
	learning_rate = 0.000001
	max_iterations = 1

	linear_regretion = mlr()
	regr = linear_model.LinearRegression()
	tools = t.tools()

	tools.loadDataset('bike_sharing.csv')
	trfeatures, trlabels, ttfeatures, ttlabels = tools.splitDataset()

	regr.fit(trfeatures, trlabels)
	linear_regretion.fit(trfeatures, trlabels, learning_rate, max_iterations)

	print('sklearn = %f' %regr.score(ttfeatures, ttlabels))
	print('self = %f' %linear_regretion.score(ttfeatures, ttlabels))

main()