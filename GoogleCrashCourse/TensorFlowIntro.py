from __future__ import print_function
import math 
from IPython import display
from matplotlib import cm, gridspec, pyplot as plt 
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf 
from tensorflow.python.data import Dataset 

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe=california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /=1000.0
california_housing_dataframe

california_housing_dataframe.describe()

my_feature=california_housing_dataframe[["total_rooms"]]
feature_columns=[tf.feature_column.numeric_column("total_rooms")]

#define targets
targets=california_housing_dataframe["median_house_value"]

#configure linear regressor
my_opt= tf.train.GradientDescentOptimizer(learning_rate=0.000001)
my_opt=tf.contrib.estimator.clip_gradients_by_norm(my_opt,5.0)
linear_regressor= tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_opt
)
#Define the input funtion
def input_fn(features,targets, batch_size=1, shuffle=True, num_epochs=None):
    features={key: np.array(val) for key,val in dict(features).items()}
    ds=Dataset.from_tensor_slices((features, targets))
    ds=ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds=ds.shuffle(buffer_size=10000)
    features,labels= ds.make_one_shot_iterator().get_next()
    return features, labels
#Train model
_ = linear_regressor.train(
        input_fn=lambda:input_fn(my_feature,targets),
        steps=100
)
#Evaluate Model
predict_input_fn=lambda: input_fn(my_feature, targets, shuffle=False, num_epochs=1)
predictions=linear_regressor.predict(input_fn=predict_input_fn)
predictions=[np.array(item['predictions'][0]) for item in predictions]
mse= metrics.mean_squared_error(predictions,targets)
r_mse= math.sqrt(mse)
print("Root mean squared error: %.3f" %r_mse)

min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value
print("Min %0.3f Max %0.3f number rooms" % (california_housing_dataframe[["total_rooms"]].min(), california_housing_dataframe[["total_rooms"]].max() ))
print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % r_mse)

#Improve model- first get overall summary
calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
calibration_data.describe()

#Visualise scatter plot
sample= california_housing_dataframe.sample(n=300)
x_0=sample["total_rooms"].min()
x_1=sample["total_rooms"].max()
weights=linear_regressor.get_variable_value("linear/linear_model/total_rooms/weights")[0]
bias=linear_regressor.get_variable_value("linear/linear_model/bias_weights")
y_0=x_0*weights+bias
y_1=x_1*weights+bias
plt.plot([x_0,y_0], [x_1,y_1], c='r')

plt.xlabel("Total number rooms")
plt.ylabel("Average house value")
plt.scatter(sample["total_rooms"], sample["median_house_value"])
plt.show()



