import sys
import csv
from naive_bayesian_classify import NaiveBayesianClassifier

# Read transaction vector output by DM1.py(Assignment 1)
def parseTM(filepath = r'transaction_matrix.csv'):
    tm = []

    with open(filepath, 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        for row in reader:
            topics = map(str.strip, row[2].split(','))
            if topics[0] == '': continue
            row[2] = topics
            row[1] = map(str.strip, row[1].split(','))
            tm.append(row)

    del tm[0]
    return tm



# Cross-validation
def cross_validate(no_of_folds):
    print "CROSS VALIDATION(Number of folds: %d) " % no_of_folds
    print " Fold  TrainTime  TestTime        A0        A1        A2"

    for i in range(no_of_folds):
        sys.stdout.write("%5d" % (i+1))
        test_start = i*len(tm)//no_of_folds
        test_end = (i+1)*len(tm)//no_of_folds
        nbc.classify(tm, test_start, test_end)

# Simply classify given the test size
def simple_classify(test_percentage):
    print " Split TrainTime  TestTime        A0        A1        A2"
    test_start = len(tm)*(100 - test_percentage)//100
    test_end = -1

    sys.stdout.write("%2d/%2d"%((100-test_percentage),test_percentage))
    nbc.classify(tm, test_start, test_end)



##### MAIN #####
tm = parseTM()
nbc = NaiveBayesianClassifier()
arg_list = sys.argv

print "A0: Accuracy by atleast one correct prediction per aticle"
print "A1: Accuracy by all correct predictions per article"
print "A2: Accuracy by \'correct predictions/total topics\' per article"
#print "A3: Accuracy by topics predicted correctly"
print ""

if arg_list[1] == '-f': cross_validate(int(arg_list[2]))
elif arg_list[1] == '-t': simple_classify(int(arg_list[2]))
else: print "Incorrect arguments"

