from google.colab import drive
drive.mount("/content/drive")

# basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

%matplotlib inline



import warnings
warnings.filterwarnings("ignore")

# sklearn supervised learning
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# make plots nicer
sns.set()

# load data
q1_data = pd.read_csv("data.csv")
pd.DataFrame.head(q1_data)

# overview
q1_data.describe()
q1_data.corr()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

X = q1_data.iloc[:, 1:]
y = q1_data['z']


# split data into predictors and target
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
# split data into train and test set
for train_index, test_index in sss.split(X, y):
      X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
      y_train, y_test = y[train_index], y[test_index]

# perform mean imputation, could do regression imputation mice package as well...
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_train)

X_train = imp.transform(X_train)
X_test = imp.transform(X_test)

#EDA
y_train = y_train.reset_index(drop=True)
X_df = pd.DataFrame(X_train)
X_df.columns = q1_data.columns[1:]
train_df = pd.concat([y_train, X_df], axis=1)
train_df.columns = q1_data.columns
train_df.describe()
corr_mat = train_df.corr()

# plot histograms
fig, axis = plt.subplots(7,4,figsize=(30, 30))
counter = 1
for ax in axis.flatten():
    sns.histplot(train_df, x=train_df.columns[counter], ax=ax, bins=100, hue='z')
    counter += 1
    
# train RF
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.arange(2, 300, 6)]
# Number of features to consider at every split
max_features = [int(x) for x in np.arange(2, 28, 3)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.arange(1, 30, 1)]
max_depth.append(None)

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier(random_state = 0)
# Random search of parameters, using 5 fold cross validation, 
# search across 200 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 200, scoring='accuracy', 
                              cv = 5, verbose=2, random_state=0, n_jobs=-1,
                              return_train_score=True)

# Fit the random search model
rf_random.fit(X_train, y_train);

print(rf_random.best_params_)
print(rf_random.best_estimator_)
print(rf_random.best_score_)







# predictions for random forest and test accuracy
rf_rand = RandomForestClassifier(n_estimators= 146, max_features=5, max_depth= 7, random_state=0)
rf_rand.fit(X_train, y_train)
rf_pred = rf_rand.predict(X_test)
print('The accuracy obtained on the unseen data in the test set is:',accuracy_score(y_test, rf_pred))

# training neural net
# for a great tutorial on bayesian optimisation to tune hyperparameters follow the link below:
# https://colab.research.google.com/github/Hvass-Labs/TensorFlow-Tutorials/blob/master/19_Hyper-Parameters.ipynb
import tensorflow as tf
import math
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
#from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout, Dense, Flatten, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers

! pip install h5py scikit-optimize
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args

dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                         name='learning_rate')
dim_dropout_rate = Real(low=1e-2, high=0.5, prior='log-uniform',
                         name='dropout_rate')
dim_num_dense_layers = Integer(low=1, high=8, name='num_dense_layers')
dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')
dim_activation = Categorical(categories=['relu', 'sigmoid'],
                             name='activation')
dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_activation,
              dim_dropout_rate]
default_parameters = [1e-5, 1, 300, 'relu', 1e-1]

# split training set again into train and validations set
train_full_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_full_dataset = train_full_dataset.shuffle(353)
train_full_dataset = train_full_dataset.batch(89)
train_full_dataset = train_full_dataset.prefetch(tf.data.experimental.AUTOTUNE)

trainval = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in trainval.split(X_train, y_train):
      X_t, X_val = X_train[train_index,:], X_train[test_index, :]
      y_t, y_val = y_train[train_index], y_train[test_index]

# Load the data into training and validation Dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)) # should be validation data

train_dataset = train_dataset.shuffle(353)

train_dataset = train_dataset.batch(89)
val_dataset = val_dataset.batch(len(y_val)) # just one batch for validation

train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

def log_dir_name(learning_rate, num_dense_layers,
                 num_dense_nodes, activation, dropout_rate):

    # The dir-name for the TensorBoard log-dir.
    s = "./19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}_dr_{0:.0e}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       num_dense_layers,
                       num_dense_nodes,
                       activation,
                       dropout_rate)

    return log_dir
    

l2_coeff = 1e-5


def create_model(learning_rate, num_dense_layers,
                 num_dense_nodes, activation, dropout_rate):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """
    
    # Start construction of a Keras Sequential model.
    model = Sequential()

    # Add an input layer which is similar to a feed_dict in TensorFlow.
    # Note that the input-shape must be a tuple containing the image-size.

    model.add(InputLayer(input_shape = (X_train.shape[-1],)))# adjust this
    #model.add(Flatten()) #think don't need this

    # Add fully-connected / dense layers.
    # The number of layers is a hyper-parameter we want to optimize.
    for i in range(num_dense_layers):


        # Name of the layer. This is not really necessary
        # because Keras should give them unique names.
        name = 'layer_dense_{0}'.format(i+1)

        # Add the dense / fully-connected layer to the model.
        # This has two hyper-parameters we want to optimize:
        # The number of nodes and the activation function.
        model.add(Dense(num_dense_nodes,
                        activation=activation, 
                        kernel_regularizer=regularizers.l2(l2_coeff),
                        name=name))
        # Add Dropout Layer
        name = 'layer_dropout_{0}'.format(i+1)
        model.add(Dropout(dropout_rate, name=name))
        
        

    # Last fully-connected / dense layer with sigmoid-activation
    # for use in classification.
    model.add(Dense(1, activation='sigmoid')) # 2 classes
    
    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adam(lr=learning_rate)
    
    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    
    return model
    

path_best_model = '19_best_model.h5'
best_accuracy = 0.0
@use_named_args(dimensions=dimensions)

def fitness(learning_rate, num_dense_layers,
            num_dense_nodes, activation, dropout_rate):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('activation:', activation)
    print('dropout rate: {0:.1e}'.format(dropout_rate))
    print()
    
    # Create the neural network with these hyper-parameters.
    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation, dropout_rate=dropout_rate)

    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(learning_rate, num_dense_layers,
                           num_dense_nodes, activation, dropout_rate)
    
    # Create a callback-function for Keras which will be
    # run after each epoch has ended during training.
    # This saves the log-files for TensorBoard.
    # Note that there are complications when histogram_freq=1.
    # It might give strange errors and it also does not properly
    # support Keras data-generators for the validation-set.
    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False)
    
    # callback_ES = EarlyStopping(monitor = 'val_accuracy', patience=10)# probs no need for this as only 3 epochs
   
    # Use Keras to train the model.
    history = model.fit(train_dataset,
                        epochs=10,
                        #batch_size=128,
                        validation_data=val_dataset,
                        callbacks=[callback_log])

    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = history.history['val_binary_accuracy'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy:
        # Save the new model to harddisk.
        model.save(path_best_model)
        
        # Update the classification accuracy.
        best_accuracy = accuracy

    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    
    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -accuracy
    
    
fitness(x=default_parameters)
plot_convergence(search_result)
search_result.x

from sklearn.metrics import confusion_matrix

confusion_matrix_validation = confusion_matrix(y_test, np.array(rf_pred))

df_cm = pd.DataFrame(confusion_matrix_validation, columns=np.unique(y_test), index = np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
plt.title('Test Data RF')
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
print(classification_report(y_test, rf_pred))


final_model = create_model(learning_rate=0.0053,
                         num_dense_layers=1,
                         num_dense_nodes=302,
                         activation='relu', dropout_rate=0.3)

final_model.summary()
callback_ES = EarlyStopping(monitor = 'val_binary_accuracy', patience=10)

# Use Keras to train the model.
history = final_model.fit(train_full_dataset,
                    epochs=50,
                    #batch_size=128,
                    validation_data=val_dataset,
                    callbacks=[])

# Get the classification accuracy on the validation-set
# after the last training-epoch.
accuracy = history.history['val_binary_accuracy'][-1]

# Load the data into training and validation Dataset objects
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

test_dataset = test_dataset.batch(len(y_test))
final_model.evaluate(test_dataset)
mlp_pred = final_model.predict(test_dataset)
mlp_pred[mlp_pred>0.5] = 1
mlp_pred[mlp_pred<0.5] = 0

confusion_matrix_validation = confusion_matrix(y_test, np.array(mlp_pred))

df_cm = pd.DataFrame(confusion_matrix_validation, columns=np.unique(y_test), index = np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
plt.title('Test Data MLP')
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
print(classification_report(y_test, mlp_pred))

# function to compute mc nemar's test statistics
def mcnemar(pred_A, pred_B, y_true):
  errors_A = (pred_A != y_true)
  errors_B = (pred_B != y_true)

  n_A = np.sum(np.invert(errors_B[errors_A]))
  n_B = np.sum(np.invert(errors_A[errors_B]))

  score = (np.abs(n_A-n_B)-1)/np.sqrt(n_A+n_B)
  return score
  
  mlp_pred =mlp_pred.reshape(-1)
  mcnemar(rf_pred, mlp_pred, y_test)
  
from scipy.stats import norm
p = 1-norm.cdf(2.1667)
print(p)

# reject option for random forest and confusion matrix.
rej = np.zeros(X_test.shape[0])
probs = rf_rand.predict_proba(X_test)
probs.shape
max_probs = np.max(probs, axis=1)
rej[1-max_probs>0.4] = -1
rej[1-max_probs<=0.4] = 1

classes = rf_rand.predict(X_test)
classes[rej==-1] = -1

y_rej = y_test[classes!=-1]
pred_rej = classes[classes!=-1]

confusion_matrix_validation = confusion_matrix(y_rej, np.array(pred_rej))

df_cm = pd.DataFrame(confusion_matrix_validation, columns=np.unique(y_rej), index = np.unique(y_rej))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
plt.title('Reject option RF')
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
print(classification_report(y_rej, pred_rej))
