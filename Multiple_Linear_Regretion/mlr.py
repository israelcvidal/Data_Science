import numpy as np

class mlr():

	def predict_value(self, row):
		y = self.coefficients[0]
		for x in range(len(row)):
			y += self.coefficients[x+1]*row[x]
		return float(y)

	# def total_error(predicted, labels):
	# 	return sum((x-y)**2 for x,y in zip(predict,labels) )
			

	def estimate_coefficient(self, features, labels, learning_rate, max_iterations):
		coef = [0.0 for i in range(len(features[0]) + 1)]

		for iteration in range(max_iterations):
			total_error = 0
			i=0
			for row in features:
				y = predict_value(row, coef)
				error = y-float(labels[i])
				total_error += pow(error,2)
				coef[0] = coef[0] - learning_rate*error
				for j in range(len(row)):
					coef[j+1] = coef[j+1] - learning_rate*error*row[j]
				i+=1
			# print(iteration, learning_rate, total_error)
		return coef

	def fit(self, features, labels, learning_rate, max_iterations):
		self.learning_rate = learning_rate
		self.max_iterations = max_iterations
		self.coefficients = estimate_coefficient( features, labels, learning_rate, max_iterations)

	def predict(self, features):
		prediction = []
		for row in range(len(features)):
			y = self.predict_value(row)
			predictions.append(y)
		return predictions

	def total_sum_of_squares(y):
	    mean_y = np.mean(y)
	    return sum((v-mean_y)**2 for v in y)

	def r_squared(actual,predicted):
	    return 1.0 - sum((y-yb)**2)/total_sum_of_squares(y)

	def score(self, features, labels):
		predictions = self.predict(features)
		
		i = 0
		for x in range(len(predictions)):
			
