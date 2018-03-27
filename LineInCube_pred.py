import functions as fc
import pandas as pd
import os
from sklearn import metrics
import tensorflow.contrib.learn as learn
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

print("start prediction")
path = "./data"
filename_compare = os.path.join(path,"output1.csv")
filename = os.path.join(path,"testing_dataset.csv")

print("reading in files")
df_test = pd.read_csv(filename,na_values=['NA','?'])
df_param = pd.read_csv("./param/param.csv",na_values=['NA','?'])

print("preprocessing data")
fc.encode_numeric_zscore(df_test,'center_to_line',df_param['mean'][0],df_param['std'][0])
fc.encode_numeric_zscore(df_test,'line_in_sphere',df_param['mean'][1],df_param['std'][1])

result = df_test['line_in_cube']
df_test.drop('line_in_cube', axis=1, inplace=True)
x = df_test.as_matrix().astype(np.float32)

model_dir = "./dnn/LineInCube"
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=x.shape[1])]
print("building network")
regressor = learn.DNNRegressor(
    model_dir= model_dir,
    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1),
    feature_columns=feature_columns,
    hidden_units=[50, 25, 10])
    #hidden_units=[2, 5, 1])

print("predict")
pred = list(regressor.predict(x, as_iterable=True))
score = np.sqrt(metrics.mean_squared_error(pred,result))
print("Final score (RMSE): {}".format(score))
final_df = pd.DataFrame(result)
diff = result-pred
final_df.insert(1,'pred',pred)
final_df.insert(2,'diff',diff)
final_df.to_csv(filename_compare, index=False)
