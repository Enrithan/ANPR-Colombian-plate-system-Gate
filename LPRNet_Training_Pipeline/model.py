import torch
import torch.nn as nn
import torch.nn.functional as F

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)

class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
            small_basic_block(ch_in=64, ch_out=128),    # 4
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1)), # 7 - halves height
            small_basic_block(ch_in=128, ch_out=256),   # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # 11
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 13
            nn.MaxPool2d(kernel_size=(3, 3), stride=(4, 1)), # 14 - quarters height
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(2, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # 22
        )

        # Global Context channels: 64 + 128 + 256 + class_num 
        self.container = nn.Conv2d(
            in_channels=448 + class_num, 
            out_channels=class_num, 
            kernel_size=(1,1), 
            stride=(1,1)
        )

    def forward(self, x):
        keep_features = list()
        
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:  # Collect outputs of specific ReLUs
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            # We want to smash the Height dimension to 1 so we can concatenate width-wise.
            # Using basic adaptive pooling for absolute safety regardless of input height shifts
            f = F.adaptive_avg_pool2d(f, (1, x.size(3)))
            
            # Normalize internal features
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean + 1e-8)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        # Sequence comes out as (Batch, Classes, 1, Width). Remove Height dimension.
        logits = x.squeeze(2)
        return logits

def build_lprnet(lpr_max_len=8, class_num=37, dropout_rate=0.5):
    """
    Builds the LPRNet model.
    36 characters + 1 CTC Blank character = 37 classes
    """
    model = LPRNet(lpr_max_len, class_num, dropout_rate)
    return model
