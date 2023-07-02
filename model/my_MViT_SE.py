import yaml
import timm
import torch
import torch.nn as nn
from einops import rearrange,repeat


class conv_block(nn.Module):
    def __init__(self, in_channels=3,out_channels=48, size=224) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv3_block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), groups=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1,1), stride=(1,1)),
            nn.BatchNorm2d(self.out_channels, eps=1e-06)
        )

        self.conv3_ave_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), groups=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1,1), stride=(1,1)),
            nn.BatchNorm2d(self.out_channels, eps=1e-06)
        )

        self.conv5_block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=(5,5), stride=(1,1), padding=(2,2), groups=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1,1), stride=(1,1)),
            nn.BatchNorm2d(self.out_channels, eps=1e-06)
        )

        # self.conv7_block = nn.Sequential(
        #     nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=(7,7), stride=(1,1), padding=(3,3)),
        #     nn.ReLU(),
        #     # nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1,1), stride=(1,1)),
        #     nn.LayerNorm((size,size), eps=1e-06, elementwise_affine=True)
        # )

        self.conv1 = nn.Conv2d(in_channels= self.in_channels, out_channels= self.out_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        # self.conv3 = nn.Conv2d(in_channels= self.in_channels, out_channels= self.out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        # self.LN = nn.LayerNorm((size,size), eps=1e-06, elementwise_affine=True)
        # self.Relu = nn.ReLU()
        self.Pooling = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

    def forward(self, x):
        # x1 = self.conv1(x)
        # x = self.LN(x1) + self.LN(self.conv3(x))
        x = self.conv3_ave_block(x) + self.conv3_block(x) + self.conv5_block(x)
        # x = x + self.conv3_block(x1)
        # x = x + self.conv5_block(x1)
         # x = self.Relu(x)
#         x = rearrange(x1, "b n h d -> b d n h ")
        x = self.Pooling(x)
        return x
    

class se1d(nn.Module):
    def __init__(self, channels, ratio) -> None:
        super().__init__()
        self.dim = channels
        self.ratio = ratio
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        self.excitation = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=self.dim//self.ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.dim//self.ratio, out_features=self.dim, bias=False),
            nn.Sigmoid()
        )
        self.BN = nn.BatchNorm2d(self.dim)
    def forward(self, x):
        b,c,_,_ = x.size()
        x1 = x
        x = self.squeeze(x)
        x = x.view(b,c)
#         x = torch.transpose(x, dim0=-1, dim1=-2)
        x = self.excitation(x)
        x=x.view(b,c,1,1)
#         x = torch.transpose(x, dim0=-1, dim1=-2)
#         x = self.BN(x*x1)
        x = x1*x.expand_as(x1)
        x = self.BN(x)
        x = torch.unsqueeze(x, dim=1)
        x = rearrange(x, 'b c1 (c2 c3) h w -> b c2 (c1 c3) h w',c2=2)
        x = x.flatten(-2).mean(-1)
        x = torch.sum(x, dim=1)
        return x


class downsampler(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 2*in_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1,1), stride=(1,1)),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
    
    def forward(self, x):
        return self.block(x)
    
class conv_blockv2(nn.Module):
    def __init__(self, in_channels=3,out_channels=48, size=224) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), groups=self.in_channels),
            nn.BatchNorm2d(self.in_channels),
        )
        self.conv3_ave = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), groups=self.in_channels),
            nn.BatchNorm2d(self.in_channels),
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        
        return self.relu(x + self.conv3(x) + self.conv3_ave(x))
    
    
class vit_head(nn.Module):
    def __init__(
        self,
        dim_in = 768,
        num_classes=1,
        dropout_rate=0.3,
        act_func="sigmoid",
    ):

        super(vit_head, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Sequential(
            nn.Linear(dim_in, dim_in*2, bias=True),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(dim_in*2, num_classes, bias=True),
            # nn.Dropout(0.3)
        )

        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # if not self.training:
        # x = self.act(x)
        return x

class mmvit_SE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.activation = {}
        # v1
        # self.conv_block1 = nn.Sequential(
        #                             conv_block(),
        #                             conv_block(in_channels=48, out_channels=96, size=112)
        #                             )
        # self.conv_block2 = conv_block(in_channels=96, out_channels=192, size=56)
        # self.conv_block3 = conv_block(in_channels=192, out_channels=384, size=28)
        # self.conv_block4 = conv_block(in_channels=384, out_channels=768, size=14)
        
        # v2
        self.datalayer = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3,3), stride=(1,1), padding=(1,1), groups=3),
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(1,1), stride=(1,1)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.conv_block1 = nn.Sequential(
                                    conv_blockv2(in_channels=48),
                                    conv_blockv2(in_channels=48),
                                    conv_blockv2(in_channels=48),
                                    conv_blockv2(in_channels=48),
                                    downsampler(in_channels=48)
                                    )
        self.conv_block2 = nn.Sequential(
                                    conv_blockv2(in_channels=96),
                                    conv_blockv2(in_channels=96),
                                    conv_blockv2(in_channels=96),
                                    downsampler(in_channels=96)
                                    )
        self.conv_block3 = nn.Sequential(
                                    conv_blockv2(in_channels=192),
                                    conv_blockv2(in_channels=192),
                                    
                                    downsampler(in_channels=192)
                                    )
        self.conv_block4 = nn.Sequential(
                                    conv_blockv2(in_channels=384),
                                    conv_blockv2(in_channels=384),
                                    downsampler(in_channels=384)
                                    )

        # Mvit
        self.Mvitv2 = timm.create_model('mvitv2_small', pretrained=True)
        self.Mvitv2.stages[0].register_forward_hook(self.get_activation('M1'))
        self.Mvitv2.stages[1].register_forward_hook(self.get_activation('M2'))
        self.Mvitv2.stages[2].register_forward_hook(self.get_activation('M3'))
        self.Mvitv2.stages[3].register_forward_hook(self.get_activation('M4'))

        #one dimensiom SE
        self.SE = se1d(channels=1536,ratio=16)
        #header
        self.head = vit_head()
    def get_activation(self, name):
        def hook(model, input, output):
            # 如果你想feature的梯度能反向传播，那么去掉 detach（）
            self.activation[name] = output
        return hook

    def forward(self, img):
        Mvit_res = self.Mvitv2(img)
        bs, b, c, d = img.shape
        # v1
        # x = self.activation['M1'][0].transpose(1,2).reshape(bs,96,56,56) + self.conv_block1(img)
        # x = self.activation['M2'][0].transpose(1,2).reshape(bs,192,28,28) + self.conv_block2(x)
        # x = self.activation['M3'][0].transpose(1,2).reshape(bs,384,14,14) + self.conv_block3(x)
        # x = self.activation['M4'][0] + self.conv_block4(x).flatten(-2).transpose(-2,-1)
        # v2
        x = self.datalayer(img)
        x = self.activation['M1'][0].transpose(1,2).reshape(bs,96,56,56) + self.conv_block1(x)
        x = self.activation['M2'][0].transpose(1,2).reshape(bs,192,28,28) + self.conv_block2(x)
        x = self.activation['M3'][0].transpose(1,2).reshape(bs,384,14,14) + self.conv_block3(x)
        MViT_out = self.activation['M4'][0].transpose(1,2).reshape(bs,768,7,7)
        CNN_out = self.conv_block4(x)
        # MViT_out = torch.unsqueeze(MViT_out, 1)
        # CNN_out = torch.unsqueeze(CNN_out, 1)
        SE_in = torch.concat((MViT_out, CNN_out),1)
        SE_out = self.SE(SE_in)
        # SE_out = torch.sum(SE_out, dim=-2)
#         # x = torch.concat((self.activation['M4'][0],self.conv_block4(x).flatten(-2).transpose(-2,-1)), 2)
#         x = x.mean(-2)
        SE_out = self.head(SE_out)
        return SE_out