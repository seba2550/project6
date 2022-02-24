"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""
from regression import (utils,logreg)
from regression.logreg import LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler


def test_gradient_update():
	# Check that your gradient is being calculated correctly
	test_model = LogisticRegression(num_feats = 6, max_iter = 10, learning_rate = 0.00001, batch_size = 12) # Use the same model parameters from main.py
	X = np.array([[1, 2, 3, 4], [1, 2, 3, 4]]) # Create a basic feature matrix and class vector to target
	y = np.array([0, 0])
	test_model.W = np.array([1, 1, 1, 1]) # Initialize weights at 1
	gradient = test_model.calculate_gradient(X, y) # Calculate the gradient for the model
	true_gradient = np.array([1, 2, 3, 4]) # This is what the gradient should be for the model on this particular situation
	assert np.allclose(gradient, true_gradient, rtol= 0.0001) # Check that they're close! 

	

def test_loss_decrease():
	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training

	# Follow the same data loading procedure, splitting, and training from main.py
	X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                    'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol', 
                                    'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.7, split_state=42)
	np.random.seed(42) # Set seed for stability
	test_model = LogisticRegression(num_feats = 6, max_iter = 10, learning_rate = 0.00001, batch_size = 12) # Use the same model parameters from main.py
	test_model.train_model(X_train, y_train, X_val, y_val)
	
	# The final loss should be less than the first loss, if the model is performing well and the loss function is correct
	first_loss = test_model.loss_history_train[0]
	final_loss = test_model.loss_history_train[-1]
	assert final_loss < first_loss

def test_weights_update():
	# Check that self.W is being updated as expected
	# and produces reasonable estimates for NSCLC classification

	X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                    'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol', 
                                    'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.7, split_state=42)

	np.random.seed(42) 
	test_model = LogisticRegression(num_feats = 6, max_iter = 10, learning_rate = 0.00001, batch_size = 12)
	test_model.W = np.array([1, 1, 1, 1, 1, 1, 1])
	init_w = np.array([1, 1, 1, 1, 1, 1, 1])
	test_model.train_model(X_train, y_train, X_val, y_val)
	assert np.allclose(test_model.W, init_w) == False # Not the most elegant implementation, but this checks that weights are updated after training and therefore different from the initial weights

def test_model_accuracy():
	# Check accuracy of model after training

	X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                    'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol', 
                                    'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.7, split_state=42)
    
	# Scale the data using the code from main.py
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform (X_val)
	np.random.seed(42) 

	test_model = logreg.LogisticRegression(num_feats = 6, max_iter = 100, learning_rate = 0.005, batch_size = 12)
	test_model.W = np.random.randn(test_model.num_feats + 1).flatten()
	test_model.train_model(X_train, y_train, X_val, y_val)
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	y_pred = test_model.make_prediction(X_val)
	y_pred_labels = [1 if i > 0.5 else 0 for i in y_pred]
	accuracy = np.sum(y_val == y_pred_labels) / len(y_val)
	assert accuracy > 0.50 # 0.50 threshold to make sure the model is better than just randomly guessing
