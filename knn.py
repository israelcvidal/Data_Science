import operator
import tools as t
class knn:
	def __init__(self, n_neighbors):
		self.k = n_neighbors

	def getNeighbors(self, trfeatures, trlabels, current, k):
		distances = []
		neighbors = []
		tools = t.tools()
		#considering that current doesnt have a label anymore
		length = len(current)
		# if one wants to pass current with label: 
		# length = len(current)-1

		for x in range(len(trfeatures)):
			dist = tools.euclidianDistance(current, trfeatures[x], length)
			distances.append((trlabels[x], dist))

		# sorting distance's list by dist
		distances.sort(key=operator.itemgetter(1))

		# getting k nearest neighbors
		for x in range(k):
			neighbors.append(distances[x][0])
		return neighbors

	def getResponse(self, neighbors):
		classVotes = {}
		for x in range(len(neighbors)):
			# getting label of neighbor
			response = neighbors[x]
			if response in classVotes:
				classVotes[response] += 1
			else:
				classVotes[response] = 1

		#getting the most voted class 
		sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
		return sortedVotes[0][0]

	def getAccuracy(self, labels, predictions):
		correct = 0
		for x in range(len(labels)):
			if labels[x] == predictions[x]:
				correct += 1
		return (correct/float(len(labels)))

	def fit(self, features, labels):
		self.trfeatures = features[:]
		self.trlabels = labels[:]

	def predict(self, ttfeatures):
		predictions = []
		for x in range(len(ttfeatures)):
			neighbors = self.getNeighbors(self.trfeatures, self.trlabels, ttfeatures[x], self.k)
			result = self.getResponse(neighbors)
			predictions.append(result)
		return predictions

	def score(self, ttfeatures, ttlabels):
		predictions = self.predict(ttfeatures)
		return self.getAccuracy(ttlabels, predictions)
		
def main():
	tools = t.tools()
	kn = knn(3)

	tools.loadDataset('haberman.data')
	trfeatures, trlabels, ttfeatures, ttlabels = tools.splitDataset()

	kn.fit(trfeatures, trlabels)

	score = kn.score(ttfeatures, ttlabels)
	print score

main()