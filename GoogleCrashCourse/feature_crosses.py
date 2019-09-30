from __future__ import print_function
import tensorflow as tf 
import pandas as pd 
from tensorflow.python.data import Dataset
import math
import numpy as np 
from IPython import display
from matplotlib import plt as pyplot, cm, gridspec
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format= "{:.1f}".format

#Load and randomise feature data
california_housing_features= pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_features.reindex(np.random.permutation(california_housing_features.index))

def preprocess_features(in_df):
    selected_features=[
     "latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]
    output_features= in_df(selected_features).copy()

    output_features["rooms_per_person"]=(
        output_features["total_rooms"]/output_features["population"]
    )
    return output_features

def preprocess_targets(in_df):
    output_targets=pd.DataFrame()
    output_targets["median_house_value"]=(
        in_df["median_house_value"]/1000.0)
    return output_targets

training_features= preprocess_features(california_housing_features.head(n=12000))
training_targets=preprocess_targets(california_housing_features.head(n=12000))
validation_features= preprocess_features(california_housing_features.tail(n=5000))
validation_targets=preprocess_targets(california_housing_features.tail(n=5000))

print("Training and validation features summary:")
display.display(training_features.describe())
display.display(validation_features.describe())
print("Training and validation targets summary:")
display.display(training_targets.describe())
display.display(validation_targets.describe())

def construct_feature_cols(df_in):
    #item here is a string
    return  set(
        [ tf.feature_column.numeric_column(item) for item in df_in]
    ) 

def my_input_fn(features,targets, batch_size=1, num_epochs=None, shuffle=True):
    
    #convert features into a dict of np arrays
    features= {key: np.array(val) for key,val in dict(features).items()}

    #construct dataset, configure batching/repeating
    ds= Dataset.from_tensor_slices((features,targets))
    ds=ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds=ds.shuffle(1000)

    features,labels= ds.make_one_shot_iterator().get_next()

    return features,labels

def train_model(
        training_examples, 
        training_targets, 
        validation_examples,
        validation_targets,
        feature_columns,
        steps,
        learning_rate,
        batch_size=1,
    ):
        periods=10
        steps_per_period=steps/periods
        
        my_optimizer=tf.train.FtrlOptimizer(learning_rate=learning_rate)
        my_optimizer=tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 0.5)
        linear_regressor= tf.estimator.LinearRegressor(
            feature_columns=feature_columns,
            optimizer=my_optimizer
        )

        training_input_fn= my_input_fn(training_examples, training_targets, batch_size=batch_size)
        predict_validation_input_fn=my_input_fn(validation_examples, validation_targets, batch_size=1, num_epochs=1)
        predict_training_input_fn=my_input_fn(training_examples, training_targets,batch_size=1, num_epochs=1 )

        training_rmse=[]
        validation_rmse=[]
        for p in range(periods):
            linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)
            
            training_predictions= linear_regressor.predict(predict_training_input_fn)
            training_predictions=np.array([item["predictions"][0] for item in training_predictions])
            validation_predictions= linear_regressor.predict(predict_validation_input_fn)
            validation_predictions=np.array([item["predictions"][0] for item in validation_predictions])

            training_root_mean_squared_error= math.sqrt(
                metrics.mean_squared_error(training_predictions, training_targets)
            )
            validation_root_mean_squared_error=math.sqrt(
                metrics.mean_squared_error(validation_targets, validation_predictions)
            )

            training_rmse.append(training_root_mean_squared_error)
            validation_rmse.append(validation_root_mean_squared_error)

        plt.ylabel("RMSE")
        plt.xlabel("Periods")
        plt.plot(training_rmse, label="training")
        plt.plot(validation_rmse, label="validation")
        plt.title("Root mean squared error vs periods")
        plt.legend()
        return linear_regressor
    
_= train_model(
    training_features,
    training_targets,
    validation_features,
    validation_targets,
    feature_columns=construct_feature_columns,
    steps=500,
    learning_rate=1.0,
    batch_size=100,    
)

def get_quantile_based_boundaries(feature_vals, num_buckets):
    boundaries= np.arange(1.0, num_buckets)/num_buckets
    quantiles=feature_vals.quantile(boundaries)
    return[quantiles[q] for q in quantiles.key()]
households=tf.feature_column.numeric_column("households")
bucketized_households=tf.feature_column.bucketized_column(
    households, boundaries=get_quantile_based_boundaries(california_housing_features["households"], 7)
)
longitude=tf.feature_column.numeric_column("longitude")
bucketized_longitude= tf.feature_column.bucketized_column(
    longitude, boundaries= get_quantile_based_boundaries(california_housing_features["longitude"], 10)
)

def construct_feature_columns():
  """Construct the TensorFlow Feature Columns.

  Returns:
    A set of feature columns
  """ 
  households = tf.feature_column.numeric_column("households")
  longitude = tf.feature_column.numeric_column("longitude")
  latitude = tf.feature_column.numeric_column("latitude")
  housing_median_age = tf.feature_column.numeric_column("housing_median_age")
  median_income = tf.feature_column.numeric_column("median_income")
  rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")
  
  # Divide households into 7 buckets.
  bucketized_households = tf.feature_column.bucketized_column(
    households, boundaries=get_quantile_based_boundaries(
      training_features["households"], 7))

  # Divide longitude into 10 buckets.
  bucketized_longitude = tf.feature_column.bucketized_column(
    longitude, boundaries=get_quantile_based_boundaries(
      training_features["longitude"], 10))


  bucketized_latitude = tf.feature_column.bucketized_column(
      latitude, boundaries= get_quantile_based_boundaries(
          training_features["latitude"],10
      )
  )
  bucketized_housing_median_age = tf.feature_column.bucketized_column(
      housing_median_age, boundaries= get_quantile_based_boundaries(
          training_features["housing_median_age"], 10
      )
  )
  bucketized_median_income =tf.feature_column.bucketized_column(
      median_income, boundaries=get_quantile_based_boundaries(
          training_features["median_income"],10
      )
  )

  bucketized_rooms_per_person = tf.feature_column.bucketized_column(
      rooms_per_person, boundaries= get_quantile_based_boundaries(
          training_features["rooms_per_person"],5
      )
  )
  long_x_lat = tf.feature_column.crossed_column(
  set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000) 
  
  
  feature_columns = set([
    bucketized_longitude,
    bucketized_latitude,
    bucketized_housing_median_age,
    bucketized_households,
    bucketized_median_income,
    bucketized_rooms_per_person,
    long_x_lat])
  
  return feature_columns








 