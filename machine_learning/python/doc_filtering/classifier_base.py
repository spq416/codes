#import regular expressions and
#math libraries
import re
import math


#general class that encapsulates the
#notion of a classifier. Classification requires
#training examples. We must be able to split the
#given set into training and testing as well as performing
#cross validation
class ClassifierBase(object):


	@staticmethod
	def get_words(doc):

		splitter = re.compile('\\W*')

		#split the words by non-alpha characters
		words = [word.lower() for word in splitter.split(doc) if len(word)>2 and len(word)<20]
		return dict([(word,1) for word in words])

	#ctor initialize the classifier by 
	#specifying how to get the features
	def __init__(self,getfeatures,filename=None):
	
		#Counts of feature/category combinations
		self.feature_category_counts={}

		#counts of documents in each category
		self.category_counts={}
		self.get_features=getfeatures

		
	#increase the count of a feature/category pair
	def increment_feature_count(self,f,cat):
		self.feature_category_counts.setdefault(f,{})
		self.feature_category_counts[f].setdefault(cat,0)
		self.feature_category_counts[f][cat] +=1

	#increase the count of a category
	def increment_category_count(self,cat):
		self.category_counts.setdefault(cat,0)
		self.category_counts[cat]+=1

	# get the number of times a feature has appeared in a category
	def get_feature_count(self,f,cat):
		if f in self.feature_category_counts and cat in self.feature_category_counts[f]:
			return float(self.feature_category_counts[f][cat])
		return 0.0

	#get the feature/category combinations
	def get_features_categories(self):
		return self.feature_category_counts


	#get the feature/category combinations as a list of tuples
	def get_features_categories_tuples(self):

		features=[]

		for f in self.feature_category_counts:
			features.append((f,self.feature_category_counts[f]))
		return features

	# The number of items in a category
	def get_category_count(self,cat):
		if cat in self.category_counts:
			return float(self.category_counts[cat])
		return 0

	#get the categories of the documents
	def get_document_categories(self):
		return self.category_counts


	# get the category counts
	def get_category_counts(self):
		return  self.category_counts.values()

	# The total number of items
	def get_total_category_counts(self):
		return sum(self.category_counts.values())

	# The list of all categories
	def get_categories(self):
		return self.category_counts.keys()

	#the train method. Takes an item and a classification for this
	#item. It uses the get_features function of the class to break the
	#item into its separate features. It then calls incr_feature to increase the
 	#the counts for this classification for every feature. Finally it increases
	#the total count for this classification
	def train(self,item,cat):

		features = self.get_features(item)

		for f in features:
			self.increment_feature_count(f,cat)

		self.increment_category_count(cat)

	#calculate conditional probability that an item is in
	#a particular category. In other words, it calculates
        #Pr(feature|category)
	def feature_conditional_probability(self,f,cat):
		if self.get_category_count(cat)==0: return 0
		return self.get_feature_count(f,cat)/self.get_category_count(cat)

	#calculate weighted probability
	def weighted_probability(self,f,cat,prf,weight=1.0,ap=0.5):

		# Calculate current probability
		basicprob=prf(f,cat)

		# Count the number of times this feature has appeared in
		# all categories
		totals=sum([self.get_feature_count(f,c) for c in self.get_categories()])

		# Calculate the weighted average
		bp=((weight*ap)+(totals*basicprob))/(weight+totals)
		return bp


#the NaiveBayes classifier
class NaiveBayes(ClassifierBase):

	#constructor
	def __init__(self,getfeatures):
		super(NaiveBayes,self).__init__(getfeatures)
	
		#thresholds the classifier is using
		self.thresholds={}

	#set the threshold value of the category
	def set_category_threshold(self,cat,t):
		self.thresholds[cat] = t

	#get the threshold for the category cat
        #if the category is not found then returns 1
	def get_category_threshold(self,cat):
		if cat not in self.thresholds: return 1.0
		return self.thresholds[cat]


	#classify the given item
	def classify(self,item,default=None):

		probs={}

		#find the category with the highest probability
		max=0.0
		for cat in self.get_categories():
			probs[cat] = self.probability(item,cat)
			if probs[cat]>max:
				max=probs[cat]
				best = cat

		#print "\tBest is %s "%(best)
		for cat in probs:
			if cat == best: continue
			if probs[cat]*self.get_category_threshold(best)>probs[best]: return default

		return best

	#get the overall probability of a document being given a classification. Extracts the features (words)
	#and multiplies all their probabilities together to get an
 	#overall probability
	def document_probability(self,item,cat):

		features = self.get_features(item)

		p=1
  		for feature in features:
			p *= self.weighted_probability(feature,cat,self.feature_conditional_probability)
		return p

	#calculate the probability of the category and returns the product
        #of Pr(document | category)*Pr(category)
	def probability(self,item,cat):
		cat_probability = self.get_category_count(cat)/self.get_total_category_counts()
		doc_probability = self.document_probability(item,cat)
		return doc_probability*cat_probability


	#train the classifier by passing the
	#words the number of docs with Good and the
	#number of docs with bad classification
	def train_from_userdata(self,thelist,ngood_docs,nbad_docs):

		for d in range(ngood_docs):
			self.increment_category_count('Good')

		for d in range(nbad_docs):
			self.increment_category_count('Bad')

		for el in range(len(thelist)):
			tpl = thelist[el]
			feature = tpl[0]
			good_rslt = tpl[1][1]
			bad_rslt = tpl[1][2]

			for g in range(good_rslt):
				self.increment_feature_count(feature,'Good')
				
			for bad in range(bad_rslt):
				self.increment_feature_count(feature,'Bad')
			


#The Fisher method, named after R.A. Fisher is an alternative method that
#can give very accurate results particularly for spam filtering. Unlike the
#NaiveBayes filter which uses the feature probabilities to create a whole
#document probability, the Fisher method calculates the probability of
#a category for each feature in the document, then combines the probabilities
#and tests to see if the set of probabilities is more or less likely than a random set
#This method also returns a probability for each category that can be compared to the others. 
#Although this is a more complex method, it is worth learning because it allows much greater flexibility when choosing cutoffs for categorization.
class FisherMethod(ClassifierBase):

	#constructor
	def __init__(self,getfeatures):
		super(FisherMethod,self).__init__(getfeatures)
		self.minimums={}

	#set the minimum for the category
	def set_minimum(self,cat,min):
		self.minimums[cat]=min

	#get the minimum for the category. Returns 0 
	#if the category is not found
	def get_minimum(self,cat):
		if cat not in self.minimums: return 0
		return self.minimums[cat]

	def cprob(self,f,cat):
		# The frequency of this feature in this category
		clf=self.fprob(f,cat)
		if clf==0: return 0
		# The frequency of this feature in all the categories
		freqsum=sum([self.fprob(f,c) for c in self.categories( )])
		# The probability is the frequency in this category divided by
		# the overall frequency
		p=clf/(freqsum)
		return p

	def fisher_probability(self,item,cat):
		# Multiply all the probabilities together
		p=1
		features=self.get_features(item)
		for f in features:
			p*=(self.weightedprob(f,cat,self.cprob))
		# Take the natural log and multiply by -2
		fscore=-2*math.log(p)
		# Use the inverse chi2 function to get a probability
		return self.inverse_chi2(fscore,len(features)*2)


	def inverse_chi2(self,chi,df):
		m = chi / 2.0
		sum = term = math.exp(-m)
		for i in range(1, df//2):
			term *= m / i
			sum += term
		return min(sum, 1.0)

	def classify(self,item,default=None):
		# Loop through looking for the best result
		best=default
		max=0.0
		for c in self.categories( ):
			p=self.fisher_probability(item,c)
			# Make sure it exceeds its minimum
			if p>self.get_minimum(c) and p>max:
				best=c
				max=p
		return best


#dump some data into the ClassifierBase for testing
#the code
def sample_train(cl):

	cl.train('Nobody owns the water.','good')
	cl.train('the quick rabbit jumps fences','good')
	cl.train('buy pharmaceuticals now','bad')
	cl.train('make quick money at the online casino','bad')
	cl.train('the quick brown fox jumps','good')

#NaiveBayes classifier test
def nb_test():

	cl = NaiveBayes(ClassifierBase.get_words)
	sample_train(cl)

	
	rslt = cl.classify('quick rabbit',default='unknown')
        print "Classification result: %s"%(rslt)

	rslt = cl.classify('quick money',default='unknown')
        print "Classification result: %s"%(rslt)

	cl.set_category_threshold('bad',3.0)
	rstl = cl.classify('quick money',default='unknown')

        print "Classification result: %s"%(rslt)

	for i in range(10): sample_train(cl)

	rslt = cl.classify('quick money',default='unknown')
	print "Classification result: %s"%(rslt)



	
