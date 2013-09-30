
preprocess:
	python DM1.py

bayesian:
	python DM2_bayesian.py -t 20
    
knn:
	python DM2_KNN.py -k 5 -t 1

clean:
	rm *.pyc *.csv
