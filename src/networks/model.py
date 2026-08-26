import copy

import torch

from networks.swin import SwinUNETR
from networks.blocks import UnetOutBlock, UnetrUpBlock


class AbscessSwinUNETR(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, swin_unetr_model: SwinUNETR, swin_in_channels=4, feature_size=48, freeze_all=True):
        super(AbscessSwinUNETR, self).__init__()
        if freeze_all:
            for child in swin_unetr_model.children():
                for param in child.parameters():
                    param.requires_grad = False
        
        self.swin_unetr = swin_unetr_model
        # self.input_conv = torch.nn.Sequential(
        #     torch.nn.Conv3d(in_channels, swin_in_channels, kernel_size=1, stride=1, padding=0, bias=True),
        #     torch.nn.InstanceNorm3d(swin_in_channels),
        #     torch.nn.GELU()
        # )
        self.swin_unetr.out = torch.nn.Conv3d(feature_size, out_channels, kernel_size=1, stride=1, bias=True)

    def unfreeze_decoders(self, decoder1=False, decoder2=False, decoder3=False, decoder4=False, decoder5=False):
        if decoder1:
            for param in self.swin_unetr.decoder1.parameters():
                param.requires_grad = True
        if decoder2:
            for param in self.swin_unetr.decoder2.parameters():
                param.requires_grad = True
        if decoder3:
            for param in self.swin_unetr.decoder3.parameters():
                param.requires_grad = True
        if decoder4:
            for param in self.swin_unetr.decoder4.parameters():
                param.requires_grad = True
        if decoder5:
            for param in self.swin_unetr.decoder5.parameters():
                param.requires_grad = True

    def unfreeze_encoders(self, encoder10=False, encoder4=False, encoder3=False, encoder2=False, encoder1=False):
        if encoder1:
            for param in self.swin_unetr.encoder1.parameters():
                param.requires_grad = True
        if encoder2:
            for param in self.swin_unetr.encoder2.parameters():
                param.requires_grad = True
        if encoder3:
            for param in self.swin_unetr.encoder3.parameters():
                param.requires_grad = True
        if encoder4:
            for param in self.swin_unetr.encoder4.parameters():
                param.requires_grad = True
        if encoder10:
            for param in self.swin_unetr.encoder10.parameters():
                param.requires_grad = True
    
    def unfreeze_transformers(self,):
        for param in self.swin_unetr.swinViT.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        for param in self.swin_unetr.parameters():
            param.requires_grad = True

    def forward(self, x):
        # x = self.input_conv(x)
        return self.swin_unetr(x)
    

class AbscessSepSwinUNETR(torch.nn.Module):
    def __init__(self, brain_out_channels: int, lesion_out_channels: int, swin_unetr_model: SwinUNETR, feature_size=48, freeze_all=True):
        super(AbscessSepSwinUNETR, self).__init__()
        if freeze_all:
            for child in swin_unetr_model.children():
                for param in child.parameters():
                    param.requires_grad = False
        
        self.swin_unetr = swin_unetr_model
        
        self.brain_decoder1 = UnetrUpBlock(
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.lesion_decoder1 = UnetrUpBlock(
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        if self.swin_unetr.decoder1 is not None:
            self.brain_decoder1.load_state_dict(copy.deepcopy(self.swin_unetr.decoder1.state_dict()))
            self.lesion_decoder1.load_state_dict(copy.deepcopy(self.swin_unetr.decoder1.state_dict()))
        else:
            print('swin_unetr decoder1 is None')
        self.swin_unetr.decoder1 = None
        self.swin_unetr.out = None
        self.brain_out = UnetOutBlock(in_channels=feature_size, out_channels=brain_out_channels)
        self.lesion_out = UnetOutBlock(in_channels=feature_size, out_channels=lesion_out_channels)

        if freeze_all:
            for child in self.children():
                for param in child.parameters():
                    param.requires_grad = False

    def unfreeze_outs(self, brain_out=False, lesion_out=False):
        if brain_out:
            for param in self.brain_out.parameters():
                param.requires_grad = True
        if lesion_out:
            for param in self.lesion_out.parameters():
                param.requires_grad = True

    def unfreeze_decoders(self, brain_decoder=False, lesion_decoder=False):
        if brain_decoder:
            for param in self.brain_decoder1.parameters():
                param.requires_grad = True
        if lesion_decoder:
            for param in self.lesion_decoder1.parameters():
                param.requires_grad = True

    def forward(self, x):
        hidden_states_out = self.swin_unetr.swinViT(x, self.swin_unetr.normalize)
        enc0 = self.swin_unetr.encoder1(x)
        enc1 = self.swin_unetr.encoder2(hidden_states_out[0])
        enc2 = self.swin_unetr.encoder3(hidden_states_out[1])
        enc3 = self.swin_unetr.encoder4(hidden_states_out[2])
        dec4 = self.swin_unetr.encoder10(hidden_states_out[4])
        dec3 = self.swin_unetr.decoder5(dec4, hidden_states_out[3])
        dec2 = self.swin_unetr.decoder4(dec3, enc3)
        dec1 = self.swin_unetr.decoder3(dec2, enc2)
        dec0 = self.swin_unetr.decoder2(dec1, enc1)
        brain_out = self.brain_decoder1(dec0, enc0) # outs [BS, feature-size, h, w, d]
        lesion_out = self.lesion_decoder1(dec0, enc0)
        brain_logits = self.brain_out(brain_out)
        lesion_logits = self.lesion_out(lesion_out)
        return brain_logits, lesion_logits


class SEBlock3D(torch.nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        self.pool = torch.nn.AdaptiveAvgPool3d(1)

        self.fc = torch.nn.Sequential(
            torch.nn.Conv3d(channels, channels // reduction, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(channels // reduction, channels, kernel_size=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class ConvBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm3d(out_ch),
            torch.nn.ReLU(),

            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm3d(out_ch),
            torch.nn.ReLU(),

            SEBlock3D(out_ch)
        )

    def forward(self, x):
        return self.block(x)


class BrainStripNet(torch.nn.Module):

    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 16)
        self.pool1 = torch.nn.MaxPool3d(2)

        self.enc2 = ConvBlock(16, 32)
        self.pool2 = torch.nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(32, 64)

        self.up2 = torch.nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)

        self.up1 = torch.nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)

        self.out_conv = torch.nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, x):

        x1 = self.enc1(x)          # 96³

        x2 = self.pool1(x1)
        x2 = self.enc2(x2)         # 48³

        x3 = self.pool2(x2)
        x3 = self.bottleneck(x3)   # 24³

        y = self.up2(x3)
        y = torch.cat([y, x2], dim=1)
        y = self.dec2(y)

        y = self.up1(y)
        y = torch.cat([y, x1], dim=1)
        y = self.dec1(y)

        return self.out_conv(y)


if __name__ == "__main__":

    model = BrainStripNet(
        in_channels=4,
        num_classes=2
    )

    x = torch.randn(2, 4, 96, 96, 96)

    y = model(x)

    print(y.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total SwinUNETR parameters count", pytorch_total_params)
    print()
