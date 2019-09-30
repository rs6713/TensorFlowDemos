from __future__ import print_function
import math
import pandas as pd 
from sklearn import metrics
import tensorflow as tf 
from tensorflow.python.data import Dataset
import numpy as np 
from matplotlib import pyplot as plt, cm, gridspec
from IPython import display
from matplotlib.ticker import MultipleLocator


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format="{:.1f}".format
cali_df=pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=',')
print(cali_df.index)
print(len(cali_df.index))
cali_df= cali_df.iloc[np.random.permutation(len(cali_df.index))]

'''
def preprocess_features(cali_df):
  selected_features = cali_df[
    ["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  processed_features["rooms_per_person"] = (
    cali_df["total_rooms"] /
    cali_df["population"])
  processed_features["distance_sanfran"]=(
       cali_df["latitude"].apply(lambda x: abs(x-38))
  )
  return processed_features
'''
def preprocess_features(cali_df):
    input_features=pd.DataFrame()
    input_features["median_income"]=cali_df["median_income"]
    for bin in range(32, 42):
        input_features["%d_%d"%(bin, bin+1)]=cali_df["latitude"].apply(
            lambda x: 1 if x>=bin and x<=bin+1 else 0
        )
    return input_features


def preprocess_targets(cali_df):
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["median_house_value"] = (
    cali_df["median_house_value"] / 1000.0)
  return output_targets


# Choose the first 12000 (out of 17000) examples for training.
training_examples = preprocess_features(cali_df.head(12000))
training_targets = preprocess_targets(cali_df.head(12000))

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = preprocess_features(cali_df.tail(5000))
validation_targets = preprocess_targets(cali_df.tail(5000))

# Double-check that we've done the right thing.
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())

correlation_df= training_examples.copy()
correlation_df["target"]= training_targets["median_house_value"]

fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlation_df.corr())
print(correlation_df.corr())
fig.colorbar(cax)
labels=[
    "median_income",
    "lat bin 1",
    "lat bin 2",
    "lat bin 3",
    "lat bin 4",
    "lat bin 5",
    "lat bin 6",
    "lat bin 7",
    "lat bin 8",
    "lat bin 9",
    "lat bin 10",
    "house price"
]
'''
labels=["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income", "rooms_per_person", "sanfran distance", "house price"]
'''
ax.set_xticklabels([''] + labels,fontsize=4)
ax.set_yticklabels([''] + labels,fontsize=4)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
plt.show()

#plt.scatter(training_examples["latitude"], training_targets["median_house_value"])
#plt.show() 

def construct_feature_columns(input_features):
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
    
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):

  periods = 10
  steps_per_period = steps / periods

  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer
  )
    
  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["median_house_value"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["median_house_value"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["median_house_value"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
    )
    # Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")

  
  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  return linear_regressor

minimal_features = [
    "34_35",
    "35_36",
    "36_37",
    "37_38",
    "38_39",
    "39_40",
    "40_41",
    "median_income",
]

assert minimal_features, "You must select at least one feature!"

minimal_training_examples = training_examples[minimal_features]
minimal_validation_examples = validation_examples[minimal_features]




train_model(
    learning_rate=0.01,
    steps=500,
    batch_size=5,
    training_examples=minimal_training_examples,
    training_targets=training_targets,
    validation_examples=minimal_validation_examples,
    validation_targets=validation_targets)
