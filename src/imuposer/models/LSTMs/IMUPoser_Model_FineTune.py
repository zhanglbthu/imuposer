r"""
IMUPoser Model
"""

import torch.nn as nn
import torch
import pytorch_lightning as pl
from imuposer.models.loss_functions import *
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix, angle_between, radian_to_degree

class IMUPoserModelFineTune(pl.LightningModule):
    r"""
    Inputs - N IMUs, Outputs - SMPL Pose params (in Rot Matrix)
    """
    def __init__(self, config, pretrained_model):
        super().__init__()
        # load a pretrained model
        self.pretrained_model = pretrained_model
        self.n_pose_output = pretrained_model.n_pose_output

        self.batch_size = config.batch_size
        self.config = config

        if config.use_joint_loss:
            self.bodymodel = ParametricModel(config.og_smpl_model_path, device=config.device)

        if config.loss_type == "mse":
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()

        self.lr = 3e-4
        self.save_hyperparameters(ignore=['pretrained_model'])

    def forward(self, imu_inputs, imu_lens):
        pred_pose = self.pretrained_model(imu_inputs, imu_lens)
        return pred_pose

    def training_step(self, batch, batch_idx):
        imu_inputs, target_pose, input_lengths, _ = batch

        _pred = self(imu_inputs, input_lengths)

        pred_pose = _pred[:, :, :self.n_pose_output]
        _target = target_pose
        target_pose = _target[:, :, :self.n_pose_output]
        loss = self.loss(pred_pose, target_pose)
        if self.config.use_joint_loss:
            pred_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[1]
            target_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose).view(-1, 216))[1] ## If training is slow, get this from the dataloader
            joint_pos_loss = self.loss(pred_joint, target_joint)
            loss += joint_pos_loss

        self.log(f"training_step_loss", loss.item(), batch_size=self.batch_size)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        imu_inputs, target_pose, input_lengths, _ = batch

        _pred = self(imu_inputs, input_lengths)

        pred_pose = _pred[:, :, :self.n_pose_output]
        _target = target_pose
        target_pose = _target[:, :, :self.n_pose_output]
        loss = self.loss(pred_pose, target_pose)
        if self.config.use_joint_loss:
            pred_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[1]
            target_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose).view(-1, 216))[1] ## If training is slow, get this from the dataloader
            joint_pos_loss = self.loss(pred_joint, target_joint)
            loss += joint_pos_loss

        self.log(f"validation_step_loss", loss.item(), batch_size=self.batch_size)

        return {"loss": loss}

    def predict_step(self, batch, batch_idx):
        imu_inputs, target_pose, input_lengths, _ = batch

        _pred = self(imu_inputs, input_lengths)

        pred_pose = _pred[:, :, :self.n_pose_output]
        _target = target_pose
        target_pose = _target[:, :, :self.n_pose_output]
        loss = self.loss(pred_pose, target_pose)
        if self.config.use_joint_loss:
            pred_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[1]
            target_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose).view(-1, 216))[1] ## If training is slow, get this from the dataloader
            joint_pos_loss = self.loss(pred_joint, target_joint)
            loss += joint_pos_loss

        return {"loss": loss.item(), "pred": pred_pose, "true": target_pose}

    def test_step(self, batch, batch_idx):

        imu_inputs, target_pose, input_lengths, _ = batch
        
        _pred = self(imu_inputs, input_lengths)
        
        pred_pose = _pred[:, :, :self.n_pose_output]
        _target = target_pose
        target_pose = _target[:, :, :self.n_pose_output]
        
        pred_pose_global, pred_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216), calc_mesh=False)
        target_pose_global, target_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose).view(-1, 216), calc_mesh=False)
        
        offset_from_p_to_t = (target_joint[:, 0] - pred_joint[:, 0]).unsqueeze(1)
        
        tre = (pred_joint - target_joint).norm(dim=2)
        jre = radian_to_degree(angle_between(pred_pose_global, target_pose_global).view(pred_pose.shape[0], -1))
        jpe = (pred_joint + offset_from_p_to_t - target_joint).norm(dim=2)
        
        return {"jre": jre, "jpe": jpe * 100}

    def training_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="train")

    def validation_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="val")

    def test_epoch_end(self, outputs):
        # self.epoch_end_callback(outputs, loop_type="test")
        jre_loss = []
        jpe_loss = []
        for output in outputs:
            jre_loss.append(output["jre"])
            jpe_loss.append(output["jpe"])
            
        # avg_loss = torch.mean(torch.Tensor(loss))
        # self.log(f"test_jpe", avg_loss, prog_bar=True, batch_size=self.batch_size)
        avg_jre = torch.mean(torch.cat(jre_loss))
        avg_jpe = torch.mean(torch.cat(jpe_loss))
        self.log(f"test_jre", avg_jre, prog_bar=True, batch_size=self.batch_size)
        self.log(f"test_jpe", avg_jpe, prog_bar=True, batch_size=self.batch_size)

    def epoch_end_callback(self, outputs, loop_type="train"):
        loss = []
        for output in outputs:
            loss.append(output["loss"])

        # agg the losses
        avg_loss = torch.mean(torch.Tensor(loss))
        self.log(f"{loop_type}_loss", avg_loss, prog_bar=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
