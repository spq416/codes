#a module that has vaious implementations
#of the k-means clustering approach and its
#variants

import numpy
from pylab import *

#import the distances calculator
from distances_computer import*

class DataPoint(object):

	def __init__(self,row_data):
		self.row_data = row_data
		self.cluster_id = -1

	#get the cluster id based on the clusters centroids
	def determine_cluster_id(self,clusters_centroids,distance):

		self.cluster_id = 0
		min_dist = distance(self.row_data,clusters_centroids[0])
		
		for c in range(1,len(clusters_centroids)):
			d = distance(self.row_data,clusters_centroids[c])
			if d - min_dist < 0.0:
				min_dist = d
				self.cluster_id = c

		return self.cluster_id

	

#a class that implements
#k-means algorithm with various implementations. Most
#methods in this class are static so that can be used without
#constructing an object. Thus, this class is basically used to
#wrap the various functions together
class KMeans(object):

	@staticmethod
	def has_converged(mu, oldmu):
    		return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))


	@staticmethod
	def get_labels(dataset,centroids,distance):

		rows_cid = []
		row_idx = 0
		for row in dataset:
			dp = DataPoint(row)
			dp_cid = dp.determine_cluster_id(centroids,distance)
			rows_cid.append((row_idx,dp_cid))
			row_idx += 1

		return rows_cid


	@staticmethod
	def get_new_centroids(dataset,rows_cid,k):
		mu = []
		sum_ = []

		for cluster in range(k):
			mu.append(numpy.zeros(dataset[0].shape))
			sum_.append(0)
		
		for r in range(len(rows_cid)):
			row_idx = rows_cid[r][0]
			row_cid = rows_cid[r][1]

			mu[row_cid] += dataset[row_idx]
			sum_[row_cid] += 1

		for i in range(k):
			mu[i] /=sum_[i]
		
		return mu 
			
	@staticmethod
	def cluster(X,k,nitrs,distance):

		
		#randomly pick an index from the
		#range of dataset and use this index to extract
		#the centroids 
		oldmu = []
		for j in range(k):
			idx = numpy.random.choice(len(X)) 
			oldmu.append(X[idx])

		clusters = None
		mu = None

		for itr in range(nitrs):
			print "Iteration %d of %d "%(itr,nitrs)

			 # Assign all points in X to clusters
			 #it returns a list with entries (row_idx,cidx)
        		clusters = KMeans.get_labels(X,oldmu,distance)

			#calculate the new centroids
			mu = KMeans.get_new_centroids(X,clusters,k)

			if KMeans.has_converged(mu,oldmu): break

			oldmu = mu

		return clusters,mu

#tests................

#test the k-means 

def test_kmeans():

	values = 1.5 * randn(10000,2)
	features = array(values)
	clusters, centroids = KMeans.cluster(features,3,1000,euclidean)
	print "Done with kmeans..."

	cdata1 = []
	cdata2 = []
	cdata3 = []

	for c in range(len(clusters)):
		cidx = clusters[c][1]
		row_idx = clusters[c][0]
		datapoint = features[row_idx]
		

		if cidx == 0:
			cdata1.append(datapoint)
		elif cidx == 1:
			cdata2.append(datapoint)
		elif cidx == 2:
			cdata3.append(datapoint)


	print "Number of points with cluster id 0 %s"%len(cdata1)
	print "Number of points with cluster id 1 %s"%len(cdata2)
	print "Number of points with cluster id 2 %s"%len(cdata3)


	fig = plt.figure()

	x1=[]
	y1=[]
	for ele in range(len(cdata1)):
		x1.append(cdata1[ele][0]) 
		y1.append(cdata1[ele][1])

	

	plot(x1,y1,'bo')

	x2=[]
	y2=[]
	for ele in range(len(cdata2)):
		x2.append(cdata2[ele][0]) 
		y2.append(cdata2[ele][1])

	plot(x2,y2,'k*')

	x3=[]
	y3=[]
	for ele in range(len(cdata3)):
		x3.append(cdata3[ele][0]) 
		y3.append(cdata3[ele][1])

	plot(x3,y3,'rs')
	show()
			
test_kmeans()



