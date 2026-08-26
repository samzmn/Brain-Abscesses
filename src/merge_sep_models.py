import os
import copy
import torch

from networks.swin import SwinUNETR
from networks.model import AbscessSepSwinUNETR

def main(
        brain_model_path: str="./runs/train/brain_out_dec1_trained_4/model_epoch_10_loss_0.0418.pt",
        lesion_model_path: str="./runs/train/lesion_out_dec1_trained_8/model_epoch_5_0.8136.pt",
        pretrained_dir: str="./pretrained_models/fold1_f48_ep300_4gpu_dice0_9059/model.pt",
        output_path: str="./runs/train/final_sep_model.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on device: {device}")

    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=False,
    )

    model_dict = torch.load(pretrained_dir, weights_only=False)["state_dict"]
    model.load_state_dict(model_dict)

    brain_model = AbscessSepSwinUNETR(
        brain_out_channels=2, 
        lesion_out_channels=4, 
        swin_unetr_model=model,
        feature_size=48, 
        freeze_all=False
    )

    lesion_model = AbscessSepSwinUNETR(
        brain_out_channels=2, 
        lesion_out_channels=4, 
        swin_unetr_model=model,
        feature_size=48, 
        freeze_all=False
    )

    checkpoint = torch.load(brain_model_path, weights_only=False)
    brain_model.load_state_dict(checkpoint["state_dict"])

    checkpoint = torch.load(lesion_model_path, weights_only=False)
    lesion_model.load_state_dict(checkpoint["state_dict"])

    lesion_model.brain_out.load_state_dict(copy.deepcopy(brain_model.brain_out.state_dict()))
    lesion_model.brain_decoder1.load_state_dict(copy.deepcopy(brain_model.brain_decoder1.state_dict()))

    state_dict = lesion_model.state_dict()
    save_dict = {"state_dict": state_dict}
    torch.save(save_dict, output_path)
    print(f"model saved to {output_path}")


if __name__ == "__main__":
    main()