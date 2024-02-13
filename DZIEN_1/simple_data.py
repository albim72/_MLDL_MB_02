import pandas as pd
import numpy as np

import tensorflow as tf




abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names = ["Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Age"]
)

print(abalone_train.head())

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop("Age")

print("_"*50)
print(abalone_features.head())

#tworzenie modelu sieci neuronowej

abalone_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(1)
])

abalone_model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam())

abalone_model.fit(abalone_features,abalone_labels,epochs=10)
