from src.utils.common import read_config
from src.utils.data_mngnt import get_data
from src.utils.model import create_model,save_model
import argparse
import os 


def training(config_path):
    config=read_config(config_path)
    validation_datasize=config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
    LOSS_FUNCTION=config["params"]["loss_function"]
    OPTIMIZER=config["params"]["optimizer"]
    METRICS=config["params"]["metrics"]
    NUM_CLASSES=config["params"]["num_class"]
    model_name=config["artifacts"]["model_name"]
    artifacts_dir=config["artifacts"]["artifacts_dir"]
    model_dir=config["artifacts"]["model_dir"]
    model_path_dir=os.path.join( artifacts_dir,model_dir)
    os.makedirs(model_path_dir,exist_ok=True)
    EPOCHS=config["params"]["epochs"]
    VALIDATION_SET=(X_valid,y_valid)
    model=create_model(LOSS_FUNCTION,OPTIMIZER,METRICS,NUM_CLASSES)
    history=model.fit(X_train,y_train,epochs=EPOCHS,validation_data=VALIDATION_SET)
    save_model(model,model_name,model_path_dir)

if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',"-c",default="config.yaml")
    parsed_args=args.parse_args()
    training(config_path=parsed_args.config)