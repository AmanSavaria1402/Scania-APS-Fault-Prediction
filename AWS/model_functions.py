import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from metric_functions import misclassification_score, plot_matrices
from datetime import datetime
import joblib
import streamlit as st

def predict(X):
	'''
		This function takes one or more datapoints, and predicts their class labels and 
		appends them to the dataframe itself
	'''
	if 'class' in X.columns:
		X.drop(columns='class', inplace=True) # dropping the class labels
	print(X.shape)

	# loading the imputation models and the imputation strategy list
	strat_list = joblib.load('Objects/strat_list.pickle')
	imputer = joblib.load("Objects/simpleImputer.pkl")

	# performing missing value imputation
	# dropping the features with high missing values
	inter_ = X.drop(columns=strat_list[3])
	inter_ = inter_.drop(columns=['cd_000'])

	# mice imputation
	data = pd.DataFrame(imputer.transform(inter_), columns=inter_.columns)


	# scaling the data
	# loading the min max scaler object
	normalizer = joblib.load('Objects/minmaxScaler.pkl')
	# normalizing the values 
	data_norm = normalizer.transform(data)

	# loading the model and the optimal threshold and predicting
	predictors = joblib.load('Objects/xgb_clf_49_77.pkl')
	model = predictors['model']
	threshold = predictors['threshold']

	# making the predictions
	# getting probabilities
	probas = model.predict_proba(data_norm)[:,1]

	# making predictions
	preds = ['pos' if p>0.2 else 'neg' for p in probas]
	data['class'] = preds

	return data