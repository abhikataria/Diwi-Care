import keras
import pandas as pd
import numpy as np
import sys
import sklearn
data = pd.read_csv('../Jupyter Work/diabetes.csv')
columns = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in columns:
    data[col].replace(0,np.NaN, inplace=True)
data.dropna(inplace = True)
dataset = data.values
X = dataset[:,0:8]
Y = dataset[:,8].astype(int)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam


scaler = StandardScaler().fit(X)
X_standardized = scaler.transform(X)
dataf = pd.DataFrame(X_standardized)
seed = 6
np.random.seed(seed)

# Start defining the model
def create_model(neuron1, neuron2):
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim = 8, kernel_initializer= 'normal', activation= 'linear'))
    model.add(Dense(2, input_dim = 16, kernel_initializer= 'normal', activation= 'linear'))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile the model
    adam = Adam(lr = 0.001)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 20, verbose = 0)

# define the grid search parameters
neuron1 = [16]
neuron2 = [2]

# make a dictionary of the grid search parameters
param_grid = dict(neuron1 = neuron1, neuron2 = neuron2)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), refit = True, verbose = 10)
grid_results = grid.fit(X_standardized, Y)

# summarize the results
#print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
data = np.array([[9,148,72,35,0,33.6,0.627,50]])
X_pred = scaler.transform(data)
pred = grid.predict(X_pred.reshape(1,-1))
print(pred)
