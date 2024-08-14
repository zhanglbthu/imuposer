# %%
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.datasets.utils import get_datamodule
from imuposer.utils import get_parser
import os

# set the random seed
seed_everything(42, workers=True)

parser = get_parser()
args = parser.parse_args()
# amass_combos: glocal+24
combo_id = args.combo_id
_experiment = args.experiment

# %%
config = Config(experiment=f"{_experiment}_{combo_id}", model="GlobalModelIMUPoserFineTuneDIP",
                project_root_dir="../../", joints_set=amass_combos[combo_id], normalize="no_translation",
                r6d=True, loss_type="mse", use_joint_loss=True, device="0",
                mkdir=False, checkpoint_path="/root/autodl-tmp/imuposer/checkpoints/IMUPoserGlobalModel_global-08102024-040836",
                test_only=True, data_dir="/root/autodl-tmp/imuposer")

# %%
# Read the best model path from the text file
with open(os.path.join(config.checkpoint_path, "best_model.txt"), "r") as f:
    # Read the first line for the best model path
    best_model_path = f.readline().strip()

# %%
# instantiate model and data
model_init = get_model(config, fine_tune=True)
model = get_model(config, pretrained=model_init)
model = model.load_from_checkpoint(best_model_path, config=config, pretrained_model=model_init)

datamodule = get_datamodule(config)
checkpoint_path = config.checkpoint_path 

# %%
wandb_logger = WandbLogger(project=config.experiment, save_dir=checkpoint_path)

trainer = pl.Trainer(logger=wandb_logger, accelerator="gpu", devices=[0], deterministic=True)

# %%
# Run the test set
trainer.test(model, datamodule=datamodule)
