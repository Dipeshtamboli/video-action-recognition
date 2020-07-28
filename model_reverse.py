import torch
import torch.nn as nn
# from mypath import Path

class C3D(nn.Module):
    """
    The C3D network.
    """
    def __init__(self, num_classes, pretrained=False):
        super(C3D, self).__init__()

        self.conv_block1=nn.Sequential(
        nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

        nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

        nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        nn.BatchNorm3d(256),
        nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

        nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

        nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        nn.BatchNorm3d(512),
        nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        )
        self.fc_block = nn.Sequential(
        nn.Linear(8192*2, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, num_classes)
        )


        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x, y):

        features_x = self.conv_block1(x)
        features_y = self.conv_block1(y)
        features_x = features_x.view(-1, 8192)
        features_y = features_y.view(-1, 8192)
        features = torch.cat((features_x,features_y), dim=1)

        logits = self.fc_block(features)
        # logits_51 = self.fc9(vid_mse_300)



        return logits


    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# def get_1x_lr_params(model):
#     """
#     This generator returns all the parameters for conv and two fc layers of the net.
#     """
#     b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
#          model.conv5a, model.conv5b, model.fc6, model.fc7]
#     for i in range(len(b)):
#         for k in b[i].parameters():
#             if k.requires_grad:
#                 yield k

# def get_10x_lr_params(model):
#     """
#     This generator returns all the parameters for the last fc layer of the net.
#     """
#     b = [model.fc8]
#     for j in range(len(b)):
#         for k in b[j].parameters():
#             if k.requires_grad:
#                 yield k

if __name__ == "__main__":
    inputs = torch.rand(10, 3, 16, 112, 112)
    # att = torch.rand(50, 300)
    net = C3D(num_classes=101, pretrained=False)

    outputs= net.forward(inputs, inputs)
    print(outputs.size())
    # print(outputs2.size())
    # print(outputs3.size())
    # print(outputs4.size())
