import torch
import os
import pickle
import numpy as np
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix, angle_between, radian_to_degree
from imuposer.config import Config

data_folder = "./data"
ckpt_name = "IMUPoserGlobalModel_global-08102024-040836"
raw_dip_path = os.path.join(data_folder, "dataset", "raw")
og_smpl_model_path = os.path.join(data_folder, "smpl", "basic_model_m.pkl")
ckpt_path = os.path.join(data_folder, "checkpoints", ckpt_name)

target_path = os.path.join(data_folder, "dataset", "processed_imuposer_25fps", "dip_test.pt")
pred_folder = os.path.join(data_folder, "results", ckpt_name, "combo")

def eval_pred(pred_pose_path, combo_id):
    body_model = ParametricModel(og_smpl_model_path, device="cuda")
    pred_pose_all = torch.load(pred_pose_path)
    target_pose = torch.load(target_path)["pose"]
    
    # check if the total frame is the same
    total_frame = 0
    for i in range(len(target_pose)):
        total_frame += target_pose[i].shape[0]
    print("total frame is", total_frame)
    assert total_frame == pred_pose_all.shape[0]
    
    # split pred_pose_all into pred_pose
    seq_len_list = [target_pose[i].shape[0] for i in range(len(target_pose))]
    pred_pose = []
    for seq_len in seq_len_list:
        pred_pose.append(pred_pose_all[:seq_len])
        pred_pose_all = pred_pose_all[seq_len:]
        
    video_folder = os.path.join(data_folder, "video", ckpt_name, combo_id)
    os.makedirs(video_folder, exist_ok=True)
    for idx in range(len(target_pose)):
        assert target_pose[idx].shape[0] == pred_pose[idx].shape[0]
        pp = pred_pose[idx]
        tp = target_pose[idx].view(-1, 216).to("cuda")
        # video_path = os.path.join(video_folder, f"motion_{idx}.mp4")
        video_path = os.path.join(video_folder, f"{idx}.mp4")
        
        body_model.view_motion([tp, pp], distance_between_subjects=0.8, video_path=video_path)

if __name__ == "__main__":
    for combo_id in os.listdir(pred_folder):
        combo_folder = os.path.join(pred_folder, combo_id)
        pred_pose_path = os.path.join(combo_folder, "pose.pt") 
        eval_pred(pred_pose_path=pred_pose_path, combo_id=combo_id)
            