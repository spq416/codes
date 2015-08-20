#the classifier
from classifier_base import*

#apache spark
from pyspark import SparkContext

#the documents we use for training
docs=[{'Good':'Hi this is Alex. How are you?'},
      {'Good':'Nobody owns the water.'},
      {'Bad':'buy pharmaceuticals now'},
      {'Good':'Hi darling this mom. I called you the other day but you did were out. Please call me as soon as possible. Bye'},
      {'Bad':'Hi darling remember me? Get in touch...'},
      {'Bad':'make quick money at the online casino'},
      {'Good':'the quick rabbit jumps fences'},
      {'Good':'the quick brown fox jumps'}]

#the doc we use for testing
test_doc = 'Hi mary whats up?'

#dummy vars indicating good and bad docs
good = -1
bad = -2


def train_map(doc):
	"""
	   return a list of elements of the form (word,(1,doc_tag))
	"""

	key = doc.keys()

	features = ClassifierBase.get_words(doc[key[0]])
	
	features_list = []

	for f in features:
		if key[0] == 'Good':
		 features_list.append((f,(features[f],-1)))
		elif key[0] == 'Bad':
		 features_list.append((f,(features[f],-2)))

	return features_list
		

def createCombiner(x):
	return [x]


def mergeValues(xs,value):
	xs.append(value)
	return xs

def mergeCombiners(x,y):
	return x+y


def reducer(x):
	"""
		x is of the form (word,[(1,doc_tag)])
		it returns a tuple of the form
		(word, [appearnces,number_of_good_appearnces,number_of_bad_appearnces])
	"""

	#how many time the word appeard
	rslt = 0

	#how many times in bad documents
	rslt_bad = 0

	#how many times in good documents
	rslt_good = 0

	for i in range(len(x[1])):
		rslt += x[1][i][0]
		if x[1][i][1] == -1:
			rslt_good +=1
		else:
			rslt_bad +=1

	return (x[0],[rslt,rslt_good,rslt_bad])


#for comparison let's run a serial version also
def serial_classifier():

	cl = NaiveBayes(ClassifierBase.get_words)

	for d in range(len(docs)):

		doc = docs[d]
		keys = doc.keys()
		
		for key in keys:
			cl.train(doc[key],key)

	best = cl.classify(test_doc,default='unknown')

	print "Serial classifier says result is %s "%best

		
if __name__ == "__main__":

	#get the SparkContext object
	sc = SparkContext(appName="Document Filtering With Apache-Spark")

	#how many threads we use
	nthrds = 2
	
       	#parallelize the documents we use for training
	training_docs_rdd = sc.parallelize(docs,nthrds)

	#collect the results of the train mapping
	training_docs_rdd_rslt = training_docs_rdd.map(train_map).collect()

	#serially merge the lists from the workers
	merged_list = training_docs_rdd_rslt[0]

	for i in range(1,len(training_docs_rdd_rslt)):
		merged_list += training_docs_rdd_rslt[i]


	#partition the merged lists
	merged_list_rdd = sc.parallelize(merged_list,nthrds)

	#combine the merged list items by keye
	merged_rslt_rdd = merged_list_rdd.combineByKey(createCombiner,mergeValues,mergeCombiners).collect()

	#finally reduce
	final_rslts_rdd = sc.parallelize(rslt_rdd,nthrds).map(reducer).collect()

	#now that we have the words and the number they appear
	#with the classification let's build the classifier
	classifier = NaiveBayes(ClassifierBase.get_words)

	classifier.train_from_userdata(final_rslts_rdd,5,3)

	#classify the test document and show the classification
	best = classifier.classify(test_doc,default='unknown')

	print "Best result is %s "%best

	serial_classifier()


        
	

		

	
		
			
		

