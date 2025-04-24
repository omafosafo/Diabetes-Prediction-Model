# Diabetes-Prediction-Model
I trained a support vector machine regressors on the Diabetes dataset in scikit-learn.  The features and target variable are already specified in the dataset. 

Data pre-processing:
I One-hot encoded sex (the second column)

Data partitioning: 
The data partitioned into 80% training and 20% test data
and Randomly shuffled before partitioning

Hyperparameter tuning: 
I tried these kernels: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’}

I Choose the best kernel, and polynomial degree  based on mean squared error from 5-fold cross-validation within the training data

Final model training and evaluation:
I trained my final SVM model with the chosen hyperparameters on the entire training data, evaluated the trained model on the test data and printed to the screen the mean squared error and mean absolute error, as well as the chosen hyperparameters
