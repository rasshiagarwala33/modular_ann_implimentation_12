import tensorflow as tf
import numpy as np
import os 
import time 

def get_timestamp(name):
    timestamp=time.asctime().replace(" ","_").replace(":","_")
    unique_name=f"{name}_{timestamp}"
    return unique_name

def get_callback_list(config,X_train):
    logs=config["logs"]
    unique_dir_name=get_timestamp("tb_logs")
    TENSORBOARD_ROOT_LOG_DIR=os.path.join(logs["logs_dir"],logs["TENSORBOARD_ROOT_LOG_DIR"],unique_dir_name)
    os.makedirs(TENSORBOARD_ROOT_LOG_DIR,exist_ok=True)
    tensorboard_cb=tf.keras.callbacks.TensorBoard(log_dir= TENSORBOARD_ROOT_LOG_DIR)
    file_writer=tf.summary.create_file_writer(logdir=TENSORBOARD_ROOT_LOG_DIR)
    with file_writer.as_default():
        images=np.reshape(X_train[10:30],(-1,28,28,1))
        tf.summary.image("20 handwritten digit samples",images,max_outputs=25,step=0)
    params=config['params']
    early_stoping_cb=tf.keras.callbacks.EarlyStopping(patience=params["patience"],restore_best_weights=params["restore_best_weights"])
    artifacts=config["artifacts"]
    CKPT_dir=os.path.join(artifacts["artifacts_dir"],artifacts["CHECKPOINTS_DIR"])
    os.makedirs(CKPT_dir,exist_ok=True)
    ckpt_path=os.path.join(CKPT_dir,"model_ckpt.h5")
    checkpoints_cb=tf.keras.callbacks.ModelCheckpoint(ckpt_path,save_best_only=True)
    return [tensorboard_cb,early_stoping_cb,checkpoints_cb]
