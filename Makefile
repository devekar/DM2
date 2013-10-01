
copy:
	cp /home/8/devekar/WWW/DM2/transaction_matrix.csv ./
	cp /home/8/devekar/WWW/DM2/data_matrix.csv ./
	cp -r /home/8/devekar/WWW/DM2/reuters ./

run_preprocess:
	python DM1.py

run_bayesian:
	python DM2_bayesian.py -t 20
    
run_knn:
	python DM2_KNN.py -k 5 -t 1

clean:
	rm *.pyc
