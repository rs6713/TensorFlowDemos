import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow.python.data import Dataset
from IPython import display
from matplotlib import pyplot as plt, cm, gridspec
import math
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format= '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

#return dataframe to use for model containing synthetic features
def preprocess_features(df_in):
    selected_features= df_in[[
        "latitude",
        "longitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"
    ]]
    processed_features=selected_features.copy()
    processed_features["rooms_per_person"]=(
        df_in["total_rooms"]/df_in["population"]
    )
    return processed_features

#returns target feature dataframe from input datafream
def preprocess_targets(df_in):
    output_targets=pd.DataFrame()
    output_targets["median_house_value"]=(
        df_in["median_house_value"]/1000.0
    )
    return output_targets

row_indexes=np.random.permutation(17000)
#choose from first 12000 examples, of 17000
training_examples= preprocess_features(california_housing_dataframe.iloc[row_indexes[:12000]])
training_targets=preprocess_targets(california_housing_dataframe.iloc[row_indexes[:12000]])
#optional .describe()
validation_examples=preprocess_features(california_housing_dataframe.iloc[row_indexes[12000:]])
validation_targets=preprocess_targets(california_housing_dataframe.iloc[row_indexes[12000:]])

plt.figure(figsize=(13,8))
ax=plt.subplot(1,2,1)
ax.set_title("Validation Data")
ax.set_autoscaley_on(False)
ax.set_ylim([32,43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(validation_examples["longitude"], validation_examples["latitude"],
    cmap="coolwarm", c=validation_targets["median_house_value"]/validation_targets["median_house_value"].max())
ax=plt.subplot(1,2,2)
ax.set_title("Training Data")
ax.set_autoscaley_on(False)
ax.set_ylim([32,43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(training_examples["longitude"], training_examples["latitude"],
    cmap="coolwarm", c=training_targets["median_house_value"]/training_targets["median_house_value"].max())
_=plt.plot()
plt.show()

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

def construct_feature_columns(input_features):
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

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
  
  # 1. Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples, training_targets, batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets, batch_size=1, num_epochs=1)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets, batch_size=1, num_epochs=1)
  
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
    # 2. Take a break and compute predictions.
    training_predictions = linear_regressor.predict(predict_training_input_fn)
    training_predictions=np.array([item['predictions'][0] for item in training_predictions])
    validation_predictions = linear_regressor.predict(predict_validation_input_fn)
    validation_predictions=np.array([item['predictions'][0] for item in validation_predictions])
    
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
    if period == (periods-1):
        plt.figure()
        plt.title("Validation Predictions vs Actual Plot")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        print(validation_targets['median_house_value'])
        print(np.array([item for item in validation_targets['median_house_value']]))
        plt.scatter(validation_predictions[:100], np.array([item for item in validation_targets['median_house_value']][:100]), c='red')
        plt.show()
        

  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()
  plt.show()
  return linear_regressor

#500
linear_regressor= train_model(0.00002, 500, 5, training_examples, training_targets, validation_examples, validation_targets)
california_housing_test_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv", sep=",")
california_features= preprocess_features(california_housing_test_data)
california_targets= preprocess_targets(california_housing_test_data)
test_input_fn= lambda: my_input_fn(california_features,california_targets["median_house_value"], num_epochs=1,batch_size=1 , shuffle=False)
test_predictions=linear_regressor.predict(test_input_fn)
test_predictions= np.array([item['predictions'][0] for item in test_predictions])
california_targets= np.array([ item for item in california_targets["median_house_value"]])
plt.figure()
plt.title("Test predictions vs targets")

plt.xlabel="Predictions"
plt.ylabel="Actual"
plt.xlim([0,500])
plt.ylim([0,500])
plt.scatter(test_predictions[:200], california_targets[:200], c='red')
plt.show()
