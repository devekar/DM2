#!/usr/bin/python
from __future__ import division
from collections import defaultdict
import itertools
import math
import timeit

# Class for K-nearest neighbor classifier
class KNN:
	word_list = []
	topic_list = []
	matrix = []
	K = 0


	# Initialize global variables
	def __init__(self, data, k):
		self.word_list = data["word_list"]
		self.topic_list = data["topic_list"]
		self.matrix = data["matrix"]	
		self.K = k
		print "K = " + str(self.K)


	# Compute magnitude of a vector
	def magnitude(self, vector):
		mag = 0;
		for item in vector:
			mag += item * item
		mag = math.sqrt(mag)
		return mag;


	# Compute cosine distance between two numeric vectors
	def cosine_dist(self, vector1, vector2):
		assert (len(vector1) == len(vector2)), \
			"Lengths of two vectors should be same"
		dot_product = 0;
		for i1, i2 in itertools.izip(vector1, vector2):
			dot_product += i1 * i2;
		mag1 = 1
		mag2 = 1
		if dot_product > 0:
			mag1 = self.magnitude(vector1);
			mag2 = self.magnitude(vector2);
		cosine_sim = dot_product / (mag1 * mag2)
		cosine_dist = 1 - cosine_sim 
		return cosine_dist


	# Get distance between two instances
	def get_dist(self, inst1, inst2):
		vector1 = inst1[1:len(self.word_list)+1]
		vector2 = inst2[1:len(self.word_list)+1]
		#print len(vector1)
		return self.cosine_dist(vector1, vector2)


	# Insert neighbor in neighbor list
	def insert_in_list(self, train_inst, distance, neighbors):
		insert_index = 0
		for index,item in enumerate(neighbors):
			if item["distance"] > distance:
				insert_index = index
				break
		if len(neighbors) == 0:
			insert_index = 0
		elif neighbors[-1]["distance"] < distance:
			insert_index = len(neighbors)
		neighbor = {}
		neighbor["id"] = train_inst[0]
		neighbor["distance"] = distance
		neighbor["classVector"] = train_inst[len(self.word_list)+1:]
		neighbors.insert(insert_index, neighbor)
		if len(neighbors) > self.K:
			neighbors.pop()

	
	# Find out if this instance is one of the neighbors so far
	# If yes, insert it into the neighbors list
	def is_neighbor(self, train_inst, distance, neighbors):
		if len(neighbors) < self.K:
			self.insert_in_list(train_inst, distance, neighbors)
		elif neighbors[-1]["distance"] > distance:
			self.insert_in_list(train_inst, distance, neighbors)

	
	# Find majority class vector in neighbors
	def majority_class(self, neighbors):
		yes_count = [0]*len(self.topic_list)
		no_count = [0]*len(self.topic_list)
		for neighbor in neighbors:
			for index,topic in enumerate(neighbor["classVector"]):
				if topic == 0:
					no_count[index] += 1
				else:
					yes_count[index] += 1
		final_class_vector = []
		for yes,no in itertools.izip(yes_count, no_count):
			if yes < no:
				final_class_vector.append(0)
			else:
				final_class_vector.append(1)
		return final_class_vector


	# Predict classes based on majority topic labels in all neighbors
	def predict_classes(self, test_inst, neighbors):
		topic_dic = defaultdict(int)
		for neighbor in neighbors:
			for index,topic_present in enumerate(neighbor["classVector"]):
				if topic_present == 1:
					topic = self.topic_list[index]
					topic_dic[topic] += 1
		sorted_topics = sorted(topic_dic, key=topic_dic.get, reverse=True)
		n = self.num_topics(test_inst[len(self.word_list)+1:])
		predicted_classes = sorted_topics[0:n]
		class_vector = []
		for topic in self.topic_list:
			if topic in predicted_classes:
				class_vector.append(1)
			else:
				class_vector.append(0)
		return class_vector


	# Classify given test instance using given training set
	def classify(self, test_inst, train_set):
		time = -timeit.default_timer()
		neighbors = []
		for train_inst in train_set:
			distance = self.get_dist(test_inst, train_inst)
			self.is_neighbor(train_inst, distance, neighbors)
			#print "distance = " + str(distance)
		#class_vector = self.majority_class(neighbors)
		class_vector = self.predict_classes(test_inst, neighbors)
		time += timeit.default_timer()
		#print "Time to classify instance = " + str(time)
		#self.topics_from_vector(class_vector)
		#self.topics_from_vector(test_inst[len(self.word_list)+1:])
		return class_vector


	def test_split(self, test_percent):
		time = -timeit.default_timer()
		test_size = int(math.floor(test_percent * len(self.matrix) // 100))
		test_set = self.matrix[:test_size]
		train_set = self.matrix[test_size:]
		print "Size of train_set = " + str(len(train_set))
		print "Size of test_set  = " + str(len(test_set))
		result_vectors = []
		for index,test_inst in enumerate(test_set):
			print "Classifying instance " + str(index+1) + " of " + str(len(test_set))
			result_vectors.append(self.classify(test_inst, train_set))
		self.compute_accuracy(result_vectors, test_set)
		time += timeit.default_timer()
		print "Time to classify = " + str(time)


	# Test the given fold and use rest of the folds as training set
	def test_fold(self, fold, nfolds):
		time = -timeit.default_timer()
		assert(fold > 0 and fold <= nfolds),"ERROR: fold number should be between 1 and " + nfolds
		print "total instances = " + str(len(self.matrix))
		fold_length = len(self.matrix) // nfolds
		#print "fold length = " + str(fold_length)
		fold_start = fold_length * (fold - 1)
		fold_end = fold_length * fold - 1
		if fold == nfolds:
			fold_end = len(self.matrix)
		#print "start = " + str(fold_start) + " end = " + str(fold_end)
		train_set = self.matrix[:fold_start]
		train_set2 = self.matrix[fold_end+1:]
		train_set = train_set + train_set2;
		test_set  = self.matrix[fold_start:fold_end+1]
		print "Size of train_set = " + str(len(train_set))
		print "Size of test_set  = " + str(len(test_set))
		result_vectors = []
		for index,test_inst in enumerate(test_set):
			print "Classifying instance " + str(index+1) + " of " + str(len(test_set))
			result_vectors.append(self.classify(test_inst, train_set))
		self.compute_accuracy(result_vectors, test_set)
		time += timeit.default_timer()
		print "Time to classify fold " + str(fold) + " = " + str(time)


	# Cross validate: make nfolds passes and test on each fold
	def cross_validate(self, nfolds):
		time = -timeit.default_timer()
		for fold in nfolds:
			self.test_fold(fold, nfolds)
		time += timeit.default_timer()
		print "Time to cross validate = " + str(time)



	# Accuracy calculation functions

	# Accuracy
	def compute_accuracy(self, results, test_set):
		self.accuracy1(results, test_set)
		self.accuracy2(results, test_set)
		self.accuracy3(results, test_set)
		mat = self.confusion_matrix(results, test_set)
		self.get_perf_metrics(mat)

    # Calculate accuracy for every instance as #correctly predicted topics / #total topics for this article
	def accuracy1(self, results, test_set):
		accuracy = 0.0
		for predicted_classes,test_inst in itertools.izip(results, test_set):
			actual_classes = test_inst[len(self.word_list)+1:]
			match_count = 0
			for predicted_class,actual_class in itertools.izip(predicted_classes,actual_classes):
				if predicted_class > 0 and actual_class > 0:
					match_count += 1
			inst_accuracy = match_count / self.num_topics(actual_classes)
			accuracy += inst_accuracy
		accuracy = accuracy * 100 / len(test_set)
		print "Accuracy 1 = " + str(accuracy) + "%"
		return accuracy

	# Calculate accuracy across test set as #correctly predicted topics / #total topics in the test set
	def accuracy2(self, results, test_set):
		match_count = 0
		total_actual_classes = 0
		for predicted_classes,test_inst in itertools.izip(results, test_set):
			actual_classes = test_inst[len(self.word_list)+1:]
			total_actual_classes += self.num_topics(actual_classes)
			for predicted_class,actual_class in itertools.izip(predicted_classes,actual_classes):
				if predicted_class > 0 and actual_class > 0:
					match_count += 1
		accuracy = match_count * 100/ total_actual_classes
		print "Accuracy 2 = " + str(accuracy) + "%"
		return accuracy

	
	# Calculate accuracy for every instance as 1 if at least one of the topics was correctly predicted
	def accuracy3(self, results, test_set):
		accuracy = 0.0
		for predicted_classes,test_inst in itertools.izip(results, test_set):
			actual_classes = test_inst[len(self.word_list)+1:]
			match = 0
			if self.num_topics(actual_classes) == 0:
				match = 1
			else:
				for predicted_class,actual_class in itertools.izip(predicted_classes,actual_classes):
					if predicted_class > 0 and actual_class > 0:
						match = 1
						break
			accuracy += match
		accuracy = accuracy * 100 / len(test_set)
		print "Accuracy 3 = " + str(accuracy) + "%"
		return accuracy


	# Get a list of topics from the class vector
	def topics_from_vector(self, class_vector):
		topics = []
		for index,item in enumerate(class_vector):
			if item > 0:
				topics.append(self.topic_list[index])
		print topics


	# Return number of topics present in this instance based on class vector
	def num_topics(self, class_vector):
		topics = 0
		for index,item in enumerate(class_vector):
			if item > 0:
				topics += 1
		return topics


	# Create confusion matrix
	def confusion_matrix(self, results, test_set):
		conf_mat = []
		for topic in self.topic_list:
			conf_mat.append(defaultdict(int))
		for predicted_classes,test_inst in itertools.izip(results, test_set):
			actual_classes = test_inst[len(self.word_list)+1:]
			i = 0
			for predicted_class,actual_class in itertools.izip(predicted_classes,actual_classes):
				if predicted_class > 0:
					if actual_class > 0:
						conf_mat[i]["TP"] += 1
					else:
						conf_mat[i]["FP"] += 1
				else:
					if actual_class > 0:
						conf_mat[i]["FN"] += 1
					else:
						conf_mat[i]["TN"] += 1
				i += 1
		#print conf_mat
		return conf_mat

	def get_perf_metrics(self, conf_mat):
		precision_vector = []
		recall_vector = []
		total_p = 0
		total_r = 0
		final_cm = defaultdict(int)
		for cm in conf_mat:
			final_cm["TP"] += cm["TP"]	
			final_cm["FP"] += cm["FP"]	
			final_cm["TN"] += cm["TN"]	
			final_cm["FN"] += cm["FN"]	
		precision = final_cm["TP"] / (final_cm["TP"] + final_cm["FP"])
		recall = final_cm["TP"] / (final_cm["TP"] + final_cm["FN"])
		fmeasure = 0
		if precision + recall > 0:
			fmeasure = 2 * precision * recall / (precision + recall)
		gmean = math.sqrt(precision * recall)
		print "Confusion Matrix: "
		print "TP = %.4f   FN = %.4f" % (final_cm["TP"], final_cm["FN"])
		print "FP = %.4f   TN = %.4f" % (final_cm["FP"], final_cm["TN"])
		print "Precision = " + str(precision)
		print "Recall    = " + str(recall)
		print "F-measure = " + str(fmeasure)
		print "G-mean    = " + str(gmean)

	def get_perf_metrics1(self, conf_mat):
		precision_vector = []
		recall_vector = []
		total_p = 0
		total_r = 0
		for cm in conf_mat:
			p = 0
			r = 0
			if cm["TP"] + cm["FP"] > 0:
				p = cm["TP"] / (cm["TP"] + cm["FP"])
				total_p += 1
			if cm["TP"] + cm["FN"] > 0:
				r = cm["TP"] / (cm["TP"] + cm["FN"])
				total_r += 1
			precision_vector.append(p)
			recall_vector.append(r)
		precision = 0.0
		recall = 0.0
		for p in precision_vector:
			precision += p
		precision = precision / total_p
		for r in recall_vector:
			recall += r
		recall = recall / total_r
		fmeasure = 0
		if precision + recall > 0:
			fmeasure = 2 * precision * recall / (precision + recall)
		gmean = math.sqrt(precision * recall)
		print "Precision = " + str(precision)
		print "Recall    = " + str(recall)
		print "F-measure = " + str(fmeasure)
		print "G-mean    = " + str(gmean)
