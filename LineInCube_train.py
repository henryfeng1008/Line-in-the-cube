import functions as fc
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow.contrib.learn as learn
import tensorflow as tf
import numpy as np
import time
import showtime as show

print("program start")
tf.logging.set_verbosity(tf.logging.ERROR)

path = "./data/"
print("reading in file")
filename_train = os.path.join(path,"training_dataset.csv")
filename_test = os.path.join(path,"testing_dataset.csv")
filename_compare = os.path.join(path,"compare.csv")
df_train = pd.read_csv(filename_train,na_values=['NA','?'])

print("preprocessing files")
mean = [df_train['center_to_line'].mean(), df_train['line_in_sphere'].mean()]
std = [df_train['center_to_line'].std(), df_train['line_in_sphere'].std()]
content = {'mean':mean,'std':std}
df_param = pd.DataFrame(data = content)
df_param.to_csv("./param/param.csv",index=False)
fc.encode_numeric_zscore(df_train,'center_to_line')
fc.encode_numeric_zscore(df_train,'line_in_sphere')


x, y = fc.to_xy(df_train , 'line_in_cube')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)

model_dir = fc.get_model_dir('LineInCube',True)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=x.shape[1])]

regressor = learn.DNNRegressor(
    model_dir= model_dir,
    #dropout=0.2,
    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1),
    feature_columns=feature_columns,
    hidden_units=[50, 25, 10])
    #hidden_units=[2, 5, 1])


validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    x_test,
    y_test,
    every_n_steps=500,
    early_stopping_metric="loss",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=50)
start_time = time.time()
print("start training")
show.now()
regressor.fit(x_train, y_train,monitors=[validation_monitor],steps=100000)

elapsed_time = time.time() - start_time
print("Elapsed time: {}".format(fc.hms_string(elapsed_time)))

print("Best step: {}, Last successful step: {}".format(
validation_monitor.best_step,validation_monitor._last_successful_step))

# Predict
pred = list(regressor.predict(x_test, as_iterable=True))
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Final score (RMSE): {}".format(score))