from imuposer.models import *

def get_model(config=None, pretrained=None, fine_tune=False):
    model = config.model
    print(model)

    # load the dataset
    if model == "GlobalModelIMUPoser" or fine_tune:
        net = IMUPoserModel(config=config)
    elif model == "GlobalModelIMUPoserFineTuneDIP":
        net = IMUPoserModelFineTune(config=config, pretrained_model=pretrained)
    else:
        print("Enter a valid model")
        return

    return net 