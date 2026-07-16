import torch

from networks.swin import SwinUNETR

class AbscessSwinUNETR(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, swin_unetr_model: SwinUNETR, swin_in_channels=4, feature_size=48):
        super(AbscessSwinUNETR, self).__init__()
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
    