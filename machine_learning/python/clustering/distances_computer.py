#a small collection of functions
#that compute distances between two given items

import math

#compute euclidean distance or equivalently
#the L2 norm between item1 and item2
def euclidean(item1,item2):

	#we have to make sure that
	#item1 and item2 have the same length
	
	if len(item1)!=len(item2):
		raise ValueError ("Length of items is not equal!!!")

	sum_ = 0
	for i in range(len(item1)):
		sum_ += (item1[i]-item2[i])*(item1[i]-item2[i])

	return math.sqrt(sum_)

#function that calculates similarity between two items
#given a list of preferences. This is a distance-based score
#for item1 and item2
def similarity_distance(prefs,item1,item2):

	#form a list of shared items
	shrd_itms = {}
	for i in prefs[item1]:
		if i in prefs[item2]:
			shrd_itms[i] = 1

	#no similarity if there are no common preferences
	if len(shrd_itms) == 0: return 0

	sum_sqrs = sum([pow(prefs[item1][item] - prefs[item2][item],2) for item in prefs[item1] if item in prefs[item2]])
	return 1./(1.+sum_sqrs)
			
	
#similarity based on Pearson correlation
def similarity_pearson(prefs, item1,item2):

	#form a list of shared items
	shrd_itms = {}
	for i in prefs[item1]:
		if i in prefs[item2]:
			shrd_itms[i] = 1

	#no similarity if there are no common preferences
	if len(shrd_itms) == 0: return 0

	#add up all the preferences
	sum1 = sum([prefs[item1][it] for it in shrd_itms])
	sum2 = sum([prefs[item2][it] for it in shrd_itms])

	#sum up the squares
	sum1Sq=sum([pow(prefs[item1][it],2) for it in shrd_itms])
	sum2Sq=sum([pow(prefs[item2][it],2) for it in shrd_itms])

	# Sum up the products
	pSum=sum([prefs[item1][it]*prefs[item2][it] for it in shrd_itms])

	# Calculate Pearson score
	n = len(shrd_itms)
	num=pSum-(sum1*sum2/n)
	den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
	if den==0: return 0
	r=num/den
	return r

#computes the pearson coefficient for the 
#two lists
def pearson(list1,list2):
	
	#compute the sums
	sum1 = sum(list1)
	sum2 = sum(list2)

	#sum of the squares
	sum1Sq=sum([pow(v,2) for v in list1])
	sum2Sq=sum([pow(v,2) for v in list2])

	# Sum of the products
	pSum=sum([list1[i]*list2[i] for i in range(len(list1))])

	# Calculate r (Pearson score)
	num=pSum-(sum1*sum2/len(list1))
	den=sqrt((sum1Sq-pow(sum1,2)/len(list1))*(sum2Sq-pow(sum2,2)/len(list1)))

	if den==0: return 0
	return 1.0-num/den




