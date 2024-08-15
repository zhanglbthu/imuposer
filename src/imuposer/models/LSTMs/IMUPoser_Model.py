r"""
IMUPoser Model
"""

import torch.nn as nn
import torch
import pytorch_lightning as pl
import json
from .RNN import RNN
from imuposer.models.loss_functions import *
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix, angle_between, radian_to_degree
from imuposer.config import Config

class IMUPoserModel(pl.LightningModule):
    r"""
    Inputs - N IMUs, Outputs - SMPL Pose params (in Rot Matrix)
    """
    def __init__(self, config:Config):
        super().__init__()
        # n_input = 12 * len(config.joints_set) # 12 for each joint(glocal is 5)
        n_input = 60

        n_output_joints = len(config.pred_joints_set) # 24
        self.n_output_joints = n_output_joints
        self.n_pose_output = n_output_joints * (6 if config.r6d == True else 9) # 24 * 6 = 144

        n_output = self.n_pose_output

        self.batch_size = config.batch_size
        
        self.dip_model = RNN(n_input=n_input, n_output=n_output, n_hidden=512, bidirectional=True)

        self.config = config
        
        self.current_combo_id = None
        self.finetuned = False
        self.results = {"combos": []}

        if config.use_joint_loss:
            self.bodymodel = ParametricModel(config.og_smpl_model_path, device=config.device)

        if config.loss_type == "mse":
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()

        self.lr = 3e-4
        self.save_hyperparameters()

    def forward(self, imu_inputs, imu_lens):
        pred_pose, _, _ = self.dip_model(imu_inputs, imu_lens)
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
        '''
        imu_inputs: (batch, 125, 12 * 5)
        target_pose: (batch, 125, 144)
        input_lengths: (batch)
        '''
        imu_inputs, target_pose, input_lengths, _ = batch

        _pred = self(imu_inputs, input_lengths) # [batch, 125, 144]

        pred_pose = _pred[:, :, :self.n_pose_output]
        _target = target_pose
        target_pose = _target[:, :, :self.n_pose_output]
        loss = self.loss(pred_pose, target_pose)
        if self.config.use_joint_loss:
            pred_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[1] # [batch * 125, 24, 3]
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
        
        '''
        pose_global: [batch * window_size, 24, 3, 3]
        joint: [batch * window_size, 24, 3]
        vertex: [batch * window_size, 6890, 3]
        '''
        pred_pose_global, pred_joint, pred_vertex = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216), calc_mesh=True)
        target_pose_global, target_joint, target_vertex = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose).view(-1, 216), calc_mesh=True)
        
        offset_from_p_to_t = (target_joint[:, 0] - pred_joint[:, 0]).unsqueeze(1)
        
        tre = (pred_joint - target_joint).norm(dim=2) # [batch * window_size, 24]
        jre = radian_to_degree(angle_between(pred_pose_global, target_pose_global).view(pred_pose_global.shape[0], -1))
        jpe = (pred_joint + offset_from_p_to_t - target_joint).norm(dim=2) # [batch * window_size, 24]
        ve = (pred_vertex + offset_from_p_to_t - target_vertex).norm(dim=2) # [batch * window_size, 6890]
        
        return {"jre": jre, "jpe": jpe * 100, "ve": ve * 100}

    def training_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="train")

    def validation_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="val")

    def test_epoch_end(self, outputs):
        # self.epoch_end_callback(outputs, loop_type="test")
        jre_loss = []
        jpe_loss = []
        ve_loss = []
        for output in outputs:
            jre_loss.append(output["jre"])
            jpe_loss.append(output["jpe"])
            ve_loss.append(output["ve"])
            
        # avg_loss = torch.mean(torch.Tensor(loss))
        # self.log(f"test_jpe", avg_loss, prog_bar=True, batch_size=self.batch_size)
        avg_jre = torch.mean(torch.cat(jre_loss))
        avg_jpe = torch.mean(torch.cat(jpe_loss))
        
        ve_sum = 0
        count = 0
        # avg_ve = torch.mean(torch.cat(ve_loss))
        # 分批计算average vertex error
        for ve in ve_loss:
            ve_sum += ve.sum()
            count += ve.numel()
        avg_ve = ve_sum / count
        
        # self.log(f"test_jre", avg_jre, prog_bar=True, batch_size=self.batch_size)
        # self.log(f"test_jpe", avg_jpe, prog_bar=True, batch_size=self.batch_size)
        # self.log(f"test_ve", avg_ve, prog_bar=True, batch_size=self.batch_size)
        
        # save the results
        current_results = {
                "combo_id": self.current_combo_id,
                "results": {
                    "jre": avg_jre.item(),
                    "jpe": avg_jpe.item(),
                    "ve": avg_ve.item()
                }
        }
        
        self.results['combos'].append(current_results)

    def epoch_end_callback(self, outputs, loop_type="train"):
        loss = []
        for output in outputs:
            loss.append(output["loss"])

        # agg the losses
        avg_loss = torch.mean(torch.Tensor(loss))
        self.log(f"{loop_type}_loss", avg_loss, prog_bar=True, batch_size=self.batch_size)

    def on_test_end(self):
        results_json_path = self.config.finetuned_model_json if self.finetuned else self.config.model_json
        with open(results_json_path, "w") as f:
            json.dump(self.results, f)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
