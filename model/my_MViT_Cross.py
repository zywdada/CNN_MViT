import yaml
import timm
import torch
import torch.nn as nn
from math import sqrt
from SCA import SCA


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
        # Softmax for evaluation and testing.
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

        return x


class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        # use mask

        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention


class self_attention(nn.Module):
    def __init__(self,hidden_size,all_head_size,head_num) -> None:
        super().__init__()
        self.hidden_size    = hidden_size       # 输入维度
        self.all_head_size  = all_head_size     # 输出维度
        self.num_heads      = head_num          # 注意头的数量
        self.h_size         = all_head_size // head_num
        
        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

    def forward(self,x):
        batch_size = x.size(0)

        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        
        attention = CalculateAttention()(q, k, v)

        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)

        output = self.linear_output(attention)

        return  output


class MHCA(nn.Module):
    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size      
        self.all_head_size  = all_head_size     
        self.num_heads      = head_num          
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        self.linear_qx = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_kx = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_vx = nn.Linear(hidden_size, all_head_size, bias=False)
        
        self.linear_qy = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_ky = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_vy = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output_x = nn.Linear(all_head_size, hidden_size)
        self.linear_output_y = nn.Linear(all_head_size, hidden_size)
        # normalization
        self.norm = sqrt(all_head_size)

    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    
    def forward(self,x,y):

        batch_size = x.size(0)

        q_y = self.linear_qx(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        k_y = self.linear_kx(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        v_y = self.linear_vx(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        
        q_x = self.linear_qy(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        k_x = self.linear_ky(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        v_x = self.linear_vy(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)



        attention_y = CalculateAttention()(q_y,k_y,v_y)
        attention_x = CalculateAttention()(q_x,k_x,v_x)
        attention_y = attention_y.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        attention_x = attention_x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        output_y = self.linear_output_y(attention_y)
        output_x = self.linear_output_x(attention_x)
        return output_x, output_y

class mmvit_cross(nn.Module):
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
                                    downsampler(in_channels=48)
                                    )
        self.conv_block2 = nn.Sequential(
                                    conv_blockv2(in_channels=96),
                                    conv_blockv2(in_channels=96),
                                    downsampler(in_channels=96)
                                    )
        self.conv_block3 = nn.Sequential(
                                    conv_blockv2(in_channels=192),
                                    conv_blockv2(in_channels=192),
                                    conv_blockv2(in_channels=192),
                                    conv_blockv2(in_channels=192),
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

        self.cross_attn = nn.ModuleList()
        for i in range(0, 3):
            self.cross_attn.append(SCA(dim=768, qk_dim=768))
        self.norm = nn.LayerNorm(768)
        self.norm_s = nn.LayerNorm(768*2)
        
        #one dimensiom SE
        # self.SE = se1d(length=768,ratio=64)
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
        # MViT_out = self.activation['M4'][0]
        # CNN_out = self.conv_block4(x).flatten(-2).transpose(-2,-1)
        MViT_out = self.activation['M4'][0].reshape(bs,7,7,768)
        CNN_out = self.conv_block4(x).flatten(-2).transpose(-2,-1).reshape(bs,7,7,768)
        for attn in self.cross_attn:
            MViT_o, CNN_o = attn(MViT_out, CNN_out)
            MViT_out = self.norm(MViT_o + MViT_out)
            CNN_out = self.norm(CNN_o + CNN_out)
            # x = torch.concat((CNN_out, MViT_out), dim=-1)
            # x = S_attn(x)
            # x = self.norm_s(x)
            # CNN_out = x[:,:,:768]
            # MViT_out = x[:,:,768:]
        x = MViT_out + CNN_out
        # x = torch.concat((self.activation['M4'][0],self.conv_block4(x).flatten(-2).transpose(-2,-1)), 2)
        x = x.mean([-2,-3])

        x = self.head(x)
        return x

# x = torch.randn(4,3,224,224)
# model = mmvit_cross()
# print(model(x))
