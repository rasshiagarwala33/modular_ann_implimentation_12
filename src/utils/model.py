import tensorflow as tf 

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):
    LAYERS=[
        tf.keras.layers.Flatten(input_shape=[28,28],name="input_layer"),
        tf.keras.layers.Dense(300,activation="relu",name="hiddenLayer1"),
        tf.keras.layers.Dense(100,activation="relu",name="hiddenLayer2"),
        tf.keras.layers.Dense(10,activation="softmax",name="outputLayer")
    ]
    model_clf=tf.keras.models.Sequential(LAYERS)
    model_clf.summary()
    model_clf.coplile(loss=LOSS_FUNCTION,optimizers=OPTIMIZER,metrics=METRICS)

    return model_clf