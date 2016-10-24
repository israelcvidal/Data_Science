import numpy as np

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
	if(len(array1) != len(array2)):
		return null
	distance = 0
	for x in range(lenght):
		distance += pow((array1[x] - array2[x]), 2)
	return math.sqrt(distance)
