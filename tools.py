# import numpy as np
import operator
import math

def loadDataset(filename):
	data = np.loadtxt(filename, delimiter="n")
	return data

def splitDataset(data):
	ndata = np.random.permutation(data)

	size = len(ndata)
	nt = int(math.floor(sizer*0.7))

	# Spliting the dataset into training and test
	trfeatures = ndata[0:nt,0:3]
	ttfeatures = ndata[nt:size, 0:3]
	trlabels = ndata[0:nt,3]
	ttlabels = ndata[nt:size, 3]

	return trfeatures, trlabels, ttfeatures, ttlabels

def euclidianDistance(array1, array2, lenght):
	# if(len(array1) != len(array2)):
	# 	return
	distance = 0
	for x in range(lenght):
		distance += pow((array1[x] - array2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(dataset, current, k):
	distances = []
	neighbors = []

	#considering that current doesnt have a label anymore
	# length = len(current)
	# if one wants to pass current with label: 
	length = len(current)-1

	for x in range(len(dataset)):
		dist = euclidianDistance(current, dataset[x], length)
		distances.append((dataset[x], dist))

	# sorting distance's list by dist
	distances.sort(key=operator.itemgetter(1))

	# getting k nearest neighbors
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		# getting label of neighbor
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1

	#getting the most voted class 
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]


def getAccuracy(dataset, predictions):
	correct = 0
	for x in range(len(dataset)):
		if dataset[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0


