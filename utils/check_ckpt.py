import torch
from collections import OrderedDict

# 加载两个 .ckpt 文件
ckpt1 = torch.load('/root/autodl-tmp/imuposer/checkpoints/IMUPoserGlobalModel_global-08102024-040836/epoch=epoch=52-val_loss=validation_step_loss=0.01043.ckpt', map_location=torch.device('cpu'))
ckpt2 = torch.load('/root/autodl-tmp/imuposer/checkpoints/IMUPoserGlobalModel_global-08102024-040836/epoch=epoch=37-val_loss=validation_step_loss=0.00463.ckpt', map_location=torch.device('cpu'))

# 提取 state_dict（模型的权重）
state_dict1 = ckpt1['state_dict']
state_dict2 = ckpt2['state_dict']

# 打印两个 .ckpt 文件的 state_dict 的 keys
# for name, param in state_dict1.items():
#     print(f"Parameter: {name}, Shape: {param.shape}")
    
new_state_dict = {}
for name, param in state_dict2.items():
    new_name = name.replace("pretrained_model.", "")
    new_state_dict[new_name] = param





# 比较两个 state_dict 是否相同
def compare_state_dicts(sd1, sd2):
    if sd1.keys() != sd2.keys():
        print("The two .ckpt files have different keys.")
        return False
    for key in sd1.keys():
        if not torch.equal(sd1[key], sd2[key]):
            print(f"Parameter {key} is different between the two .ckpt files.")
            return False
    return True

# 检查两个权重文件是否相同
are_weights_equal = compare_state_dicts(new_state_dict, state_dict1)

if are_weights_equal:
    print("The model weights in the two .ckpt files are identical.")
else:
    print("The model weights in the two .ckpt files are different.")
