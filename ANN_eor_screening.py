# Artificial Neural Network - EOR technical screening


# Part 1 - Data Pre-Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Dataset_EOR_Worldwide_2012.csv')


''' use this when testing removing all rows with missing data
dataset = dataset.iloc[:, 12:19].values
#Dropping rows with no data
from pandas import DataFrame
df = DataFrame()
dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
X = dataset.iloc[:, 3].values
y = dataset.iloc[:, 2].values
'''

X = dataset.iloc[:, 13:19].values
y = dataset.iloc[:, 12].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer.fit(X[:, :])
X[:, :] = imputer.transform(X[:, :])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



# DO NOT use when using the tuning part of this program
from keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN

#Importing the Keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from keras import losses

# Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer  
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', input_dim = 6)) # 6-nodes in the hidden layer, (average of input and output nodes, uniform intializes weights very close to 0). Activation function is  rectifier. Input dim tells there are 11 input nodes or independent variables.
#classifier.add(Dropout(p = 0.2))
#Adding the second hidden layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))
#classifier.add(Dropout(p = 0.2))
#Adding the output  layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'softmax'))
#classifier.add(Dropout(p = 0.2))
#Compiling the ANN
classifier.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) # Optimizer is stochastic gradient descent (Adam model). Loss is the Logarithmix loss function in this case (It could be the Ordinary Least squares function) 

# Fitting the ANN to the training set
history = classifier.fit(X_train, y_train, batch_size = 2, epochs = 1000)

# Plotting the accuraccy and loss
print(history.history.keys())
Max_accuracy = np.max(history.history['acc'])
Min_loss = np.min(history.history['loss'])
training_mean_accuracy = np.mean(history.history['acc'])
training_mean_loss = np.mean(history.history['loss'])

# summarize history for accuracy
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('Training accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('Training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# y_pred = (y_pred == y_pred.max(axis=1)[:,None]).astype(int) # Sets the higher value of each line to 1, and the other lines to 0
# NOT USING THIS PART, only use for binary outcomes: y_pred = (y_pred > 0.5) #returns true if y_pred is larger than 0.5. Since it's a probability we need a binary output, and hence this step.

r2_score = r2_score(y_test, y_pred)
print("r2_score: ", r2_score)
mean_absolute_error = mean_absolute_error(y_test, y_pred)
print("mean_absolute_error: ", mean_absolute_error)
mean_squared_error = mean_squared_error(y_test, y_pred)
print("mean_squared_error: ", mean_squared_error)

# Making the Confusion Matrix for binary outcomes:
# To reverse the effect of to_categorical, argmax will return the indexof the max values across axis:
y_test1 = np.argmax(y_test, axis=1)
y_pred1 = np.argmax(y_pred, axis=1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1, y_pred1)


# Testing with a diferrent dataset
# Importing the dataset
dataset1 = pd.read_csv('Dataset_EOR_Mexico.csv')
X1 = dataset1.iloc[:, 3:9].values
# Feature Scaling
from sklearn.preprocessing import StandardScaler
X1 = sc.transform(X1)
y1_pred = classifier.predict(X1)
 


# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, make_scorer, mean_squared_error
scorer = make_scorer(mean_squared_error, greater_is_better =  False)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
    #classifier.add(Dropout(p = 0.55))
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
    #classifier.add(Dropout(p = 0.55))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
    #classifier.add(Dropout(p = 0.2))
    classifier.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) # Optimizer is stochastic gradient descent (Adam model). Loss is the Logarithmix loss function in this case (It could be the Ordinary Least squares function) 
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 2, epochs = 2000)
#cv = KFold(n_splits=10, random_state = 7)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, scoring = 'accuracy', cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()



# Improving the ANN
# Dropout Regularization to reduce overfitting if needed



# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
    #classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
    #classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
    #classifier.add(Dropout(p = 0.1))
    classifier.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy']) # Optimizer is stochastic gradient descent (Adam model). Loss is the Logarithmix loss function in this case (It could be the Ordinary Least squares function) 
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs  = 100)
parameters = {'batch_size': [2, 4, 8, 16, 32],
              'optimizer': ['SGD', 'rmsprop', 'Adam', 'Adagrad', 'Adamax']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv =10)
grid_search = grid_search.fit(X_train, y_train)
best_estimator = grid_search.best_estimator_
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
Evaluate = grid_search.cv_results_

