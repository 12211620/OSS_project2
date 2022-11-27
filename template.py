#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import numpy as np
import pandas as pd
from sklearn import *

def load_dataset(dataset_path):
	df = pd.read_csv('dataset_path') 
	return df

def dataset_stat(dataset_df):	
	print(dataset_df.data.features)
	len(df.loc[df['target'] == 0])
	len(df.loc[df['target'] == 1])


def split_dataset(dataset_df, testset_size):
	X_train, X_test, y_train, y_test = train_test_split(dataset_df.data, dataset_df.target, test_size = testset_size)

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	dt_cls = DecisionTreeClassifier()
	dt_cls.fit(x_train, y_train)
	print (accuracy_score(y_test, dt_cls.predict(x_test)))
	print (precision_score(y_test, dt_cls.predict(x_test)))
	print (recall_score(y_test, dt_cls.predict(x_test)))

def random_forest_train_test(x_train, x_test, y_train, y_test):
	rf_cls = RandomForestClassifier()
	re_cls.fit(x_train, y_train)
	print (accuracy_score(rf_cls.predict(x_test),y_test))
	print (precision_score(rf_cls.predict(x_test),y_test))
	print (recall_score(rf_cls.predict(x_test),y_test))

def svm_train_test(x_train, x_test, y_train, y_test):
	svm_cls = SVC()
	svm_cls.fit(X_train, y_train)
	sv_pipe = make_pipeline(
		StandardScaler(),
		SCV
	)
	svm_pipe.fit(x_train, y_train)
	print (accuracy_score(y_test, svm_pipe.predict(x_test)))
	print (precision_score(y_test, svm_pipe.predict(x_test)))
	print (recall_score(y_test, svm_pipe.predict(x_test)))


def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)