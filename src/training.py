from src.utils.common import read_config
from src.utils.data_mngnt import get_data
from src.utils.model import create_model
import argparse


def training(config_path):
    config=read_config(config_path)
    validation_datasize=config["params"]["validation_datasize"]
    LOSS_FUNCTION=config["params"]["loss_function"]
    OPTIMIZER=config["params"]["optimizers"]
    METRICS=config["params"]["metrics"]
    NUM_CLASSES=config["params"]["no_of_classes "]
    EPOCHS=config["params"]["epochs"]
    VALIDATION_SET=(X_valid,y_valid)
    model=create_model(LOSS_FUNCTION,OPTIMIZER,METRICS,NUM_CLASSES)
    history=model.fit(X_train,y_train,epochs=EPOCHS,validation_set=VALIDATION_SET)

if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',"-c",default="config.yaml")
    parsed_args=args.parse_args()
    (X_train,y_train),(X_valid,y_valid),(X_test,y_test)=training(config_path=parsed_args.config)