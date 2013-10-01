from __future__ import division
from collections import defaultdict
import operator
import math
import timeit
import sys

class NaiveBayesianClassifier:

    def classify(self, tm, test_start, test_end):
        train_set = tm[0:test_start] + tm[test_end:]
        test_set = tm[test_start:test_end]

        start = timeit.default_timer()

        train_data =  self.train(train_set, test_set)
        train_t = timeit.default_timer()
        sys.stdout.write("%11.2f" % (train_t- start))
        sys.stdout.flush()

        accuracy = self.test(test_set, train_data[0], train_data[1], train_data[2])
        end = timeit.default_timer()
        sys.stdout.write("%10.2f" % (end- train_t))

        for i in range(3): sys.stdout.write("%10.2f" % accuracy[i])
        sys.stdout.write("\n")

        # Compute metrics from cnfusion matrix by averaging
        acc = 0
        precision = 0; precision_denom = 0
        recall = 0; recall_denom = 0
        f_measure = 0; f_measure_denom = 0

        cfm = [0,0,0,0]
        for i in self.cfm:
            for j in range(len(self.cfm[i])):
                cfm[j] += self.cfm[i][j]


        for i in self.cfm:
            acc += (self.cfm[i][0] + self.cfm[i][3])/len(test_set)
            if (self.cfm[i][0] + self.cfm[i][2]) > 0:
                precision += self.cfm[i][0]/(self.cfm[i][0] + self.cfm[i][2])
                precision_denom += 1
            if (self.cfm[i][0] + self.cfm[i][1]) > 0:
                recall += self.cfm[i][0]/(self.cfm[i][0] + self.cfm[i][1])
                recall_denom += 1
            if (2*self.cfm[i][0] + self.cfm[i][1] + self.cfm[i][2]) > 0:
                f_measure += 2*self.cfm[i][0] / (2*self.cfm[i][0] + self.cfm[i][1] + self.cfm[i][2])
                f_measure_denom += 1

        print "Accuracy:  ", (cfm[0]+cfm[3])/(cfm[0]+cfm[1]+cfm[2]+cfm[3])
        print "Precision: ", (cfm[0])/(cfm[0]+cfm[2])
        print "Recall:    ", (cfm[0])/(cfm[0]+cfm[1])
        print "F-measure: ", (2*cfm[0])/(2*cfm[0]+cfm[1]+cfm[2])
        print "G-mean:    ", math.sqrt((cfm[0])/(cfm[0]+cfm[2])*(cfm[0])/(cfm[0]+cfm[1]))
        print ""


    # Train and precompute parameters in probability calculations
    def train(self, train_set, test_set):
        topic_word = defaultdict(dict)
        topic_count = defaultdict(int)
        word_set = set()

        for article in train_set:
            for topic in article[2]:
                topic_count[topic] += 1

            for word in article[1]:
                word_set.add(word)
                for topic in article[2]:
                    topic_word[topic][word] = topic_word[topic].get(word, 0) + 1

        # Laplace Probabilities for topics
        denom = {}
        for t_key in topic_count:
            words_sum = 0
            for value in topic_word[t_key].itervalues():
                words_sum += value
            denom[t_key] = words_sum + len(word_set)

        # Laplace correction
        for key1 in topic_word:
            for key2 in topic_word[key1]:
                topic_word[key1][key2] = topic_word[key1][key2] + 1

        return [topic_word] + [denom] + [topic_count]


    # Test on the data set 
    def test(self,test_set, topic_word, denom, topic_count):
        accuracy = [0,0,0,0]
        total = 0

        self.cfm = {}
        for i in topic_count:
            self.cfm[i] = [0,0,0,0]

        for article in test_set:
            topic_P = {}

            for t_key in topic_count:
                topic_P[t_key] = topic_count[t_key]
                for word in article[1]:
                    if word in topic_word[t_key]: topic_P[t_key] *= (topic_word[t_key][word]/denom[t_key])
                    else: topic_P[t_key] *= (1/denom[t_key])

            right_predictions = self.verify_accuracy(article[2], topic_P)

            if right_predictions: accuracy[0] += 1
            if right_predictions==len(article[2]): accuracy[1] += 1
            accuracy[2] += ( right_predictions/len(article[2]) )
            accuracy[3] += right_predictions
            total += len(article[2])

        accuracy[0] = accuracy[0]/len(test_set) * 100
        accuracy[1] = accuracy[1]/len(test_set) * 100
        accuracy[2] = accuracy[2]/len(test_set) * 100
        accuracy[3] = accuracy[3]/total * 100
        return accuracy

    # Compute confusion matrix for each topic and other ways of accuracy
    def verify_accuracy( self, topics, topic_P):
        sorted_topic_P = sorted(topic_P.iteritems(), key=operator.itemgetter(1), reverse=True)
        topic_predicted = []
        for t in sorted_topic_P[:len(topics)]:
            topic_predicted.append(t[0])
        
        right_predictions = 0
        for t in topic_P:
            if t in topics and t in topic_predicted:
                self.cfm[t][0] += 1
                right_predictions += 1
            elif t in topics and t not in topic_predicted:
                self.cfm[t][1] += 1
            elif t not in topics and t in topic_predicted:
                self.cfm[t][2] += 1
            else:
                self.cfm[t][3] += 1

        return right_predictions
