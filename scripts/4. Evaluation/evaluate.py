# %%
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.datasets.utils import get_datamodule
from imuposer.utils import get_parser
import os
import sys

# set the random seed
seed_everything(42, workers=True)

parser = get_parser()
args = parser.parse_args()
# amass_combos: glocal+24
combo_id = args.combo_id
_experiment = args.experiment

# %%
config = Config(experiment=f"{_experiment}_{combo_id}", model="GlobalModelIMUPoser",
                project_root_dir="../../", joints_set=amass_combos[combo_id], normalize="no_translation",
                r6d=True, loss_type="mse", use_joint_loss=True, device="0",
                mkdir=False, checkpoint_path="/root/autodl-tmp/imuposer/checkpoints/IMUPoserGlobalModel_global-08102024-040836",
                test_only=True, data_dir="/root/autodl-tmp/dataset")

# modify batch size
config.batch_size = 1

# %%
# Read the best model path from the text file
with open(os.path.join(config.checkpoint_path, "best_model.txt"), "r") as f:
    best_model_path = f.readline().strip()
    
with open(os.path.join(config.checkpoint_path, "best_model_finetuned.txt"), "r") as f:
    best_model_finetuned_path = f.readline().strip()

ckpt = torch.load(best_model_finetuned_path)
state_dict = ckpt['state_dict']
keys_to_modify = [key for key in state_dict.keys() if key.startswith('pretrained_model.')]
if len(keys_to_modify) > 0:
    for key in keys_to_modify:
        new_key = key.replace("pretrained_model.", "")
        state_dict[new_key] = state_dict.pop(key)
        
    torch.save(ckpt, best_model_finetuned_path)
else:
    print("No keys to modify in the state_dict.")

# %%
# instantiate model and data
model = get_model(config, fine_tune=True)
model_finetuned = get_model(config, fine_tune=True)

model = model.load_from_checkpoint(best_model_path, config=config)
model_finetuned = model_finetuned.load_from_checkpoint(best_model_finetuned_path, config=config)
model_finetuned.finetuned = True

checkpoint_path = config.checkpoint_path 

# %%
wandb_logger = WandbLogger(project=config.experiment, save_dir=checkpoint_path)

trainer = pl.Trainer(logger=wandb_logger, accelerator="gpu", devices=[0], deterministic=True)

# %%
# Run the test set
combos = amass_combos
combos = {'lw_rp_h': [0, 3, 4]}
for combo_id in combos.keys():
    print(f"Running test for combo_id: {combo_id}")
    datamodule = get_datamodule(config, combo_id)
    model.current_combo_id = combo_id
    model_finetuned.current_combo_id = combo_id   

    # trainer.test(model, datamodule=datamodule)
    trainer.test(model_finetuned, datamodule=datamodule)