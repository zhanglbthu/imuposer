# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.datasets.utils import get_datamodule
from imuposer.utils import get_parser
from pathlib import Path
import os

# set the random seed
seed_everything(42, workers=True)

parser = get_parser()
args = parser.parse_args()
# amass_combos: glocal+24
combo_id = args.combo_id
fast_dev_run = args.fast_dev_run
_experiment = args.experiment

# %%
config = Config(experiment=f"{_experiment}_{combo_id}", model="GlobalModelIMUPoserFineTuneDIP",
                project_root_dir="../../", joints_set=amass_combos[combo_id], normalize="no_translation",
                r6d=True, loss_type="mse", use_joint_loss=True, device="0", 
                checkpoint_path="/root/autodl-tmp/imuposer/checkpoints/IMUPoserGlobalModel_global-08102024-040836",
                mkdir=False,
                data_dir="/root/autodl-tmp/imuposer") 

with open(os.path.join(config.checkpoint_path, "best_model.txt"), "r") as f:
    # Read the first line for the best model path
    best_model_path = f.readline().strip()

# load the pretrained model using pytorch lightning
pretrained_model = get_model(config, fine_tune=True)
pretrained_model = pretrained_model.load_from_checkpoint(best_model_path, config=config)

# %%
# instantiate model and data
model = get_model(config, pretrained_model)
datamodule = get_datamodule(config)
checkpoint_path = config.checkpoint_path 

# %%
wandb_logger = WandbLogger(project=config.experiment, save_dir=checkpoint_path)

early_stopping_callback = EarlyStopping(monitor="validation_step_loss", mode="min", verbose=False,
                                        min_delta=0.00001, patience=5)
checkpoint_callback = ModelCheckpoint(monitor="validation_step_loss", mode="min", verbose=False, 
                                      save_top_k=5, dirpath=checkpoint_path, save_weights_only=True, 
                                      filename='epoch={epoch}-val_loss={validation_step_loss:.5f}')

trainer = pl.Trainer(fast_dev_run=fast_dev_run, logger=wandb_logger, max_epochs=1000, accelerator="gpu", devices=[0],
                     callbacks=[early_stopping_callback, checkpoint_callback], deterministic=True)

# %%
trainer.fit(model, datamodule=datamodule)

# %%
# 判断checkpoint path是否为Path对象
if not isinstance(checkpoint_path, Path):
    checkpoint_path = Path(checkpoint_path)
with open(checkpoint_path / "best_model_finetuned.txt", "w") as f:
    f.write(f"{checkpoint_callback.best_model_path}\n\n{checkpoint_callback.best_k_models}")
