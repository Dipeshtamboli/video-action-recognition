import pdb
import torch
import torch.nn as nn
# from mypath import Path

def print_f(some_string, print_yes=0):
    # print_yes = 1
    if print_yes:
        print(some_string)
    else:
        pass

class C3D(nn.Module):
    """
    The C3D network.
    """
    def __init__(self, num_classes, pretrained=False):
        super(C3D, self).__init__()

        self.attn_conv2_channel = nn.Conv3d(128, 128, kernel_size=(8, 28, 28), padding=(0, 0, 0))
        self.attn_conv2_channel_conv = nn.Conv3d(256, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.attn_conv2_temp = nn.Conv3d(128, 1, kernel_size=(3, 28, 28), padding=(1, 0, 0))
        self.attn_conv2_temp_conv = nn.Conv3d(128, 128, kernel_size=(9, 3, 3), padding=(0, 1, 1))

        self.attn_conv4_channel = nn.Conv3d(512, 512, kernel_size=(2, 7, 7), padding=(0, 0, 0))
        self.attn_conv4_channel_conv = nn.Conv3d(1024, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.attn_conv4_temp = nn.Conv3d(512, 1, kernel_size=(3, 7, 7), padding=(1, 0, 0))
        self.attn_conv4_temp_conv = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(0, 1, 1))

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        # self.fc8 = nn.Linear(4096, 300)
        # self.fc9 = nn.Linear(300, 51)


        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.batchnorm1 = nn.BatchNorm3d(256)
        self.batchnorm2 = nn.BatchNorm3d(512)
        self.full_batchnorm1 = nn.BatchNorm1d(4096)
        self.full_batchnorm2 = nn.BatchNorm1d(4096)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):

        # print_f("x.shape",x.shape)
        # print_f("att.shape",att.shape)
        print_f("Input shape: {}".format(x.shape))

        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        print_f("conv1 out shape: {}".format(x.shape))

        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        print_f("conv2 out shape: {}".format(x.shape))

        attn_x = self.relu(self.attn_conv2_temp(x))
        sig_attn = torch.sigmoid(attn_x)
        print_f("conv2 attn out shape: {}".format(attn_x.shape))
        x = torch.cat((x,x*sig_attn),2)
        x = self.relu(self.attn_conv2_temp_conv(x))
        # pdb.set_trace()

        attn_x = self.relu(self.attn_conv2_channel(x))
        sig_attn = torch.sigmoid(attn_x)
        print_f("conv2 attn out shape: {}".format(attn_x.shape))
        x = torch.cat((x,x*sig_attn),1)
        x = self.relu(self.attn_conv2_channel_conv(x))
        print_f("input after conv2 attn shape: {}".format(x.shape))

        x = self.relu(self.conv3a(x))
        x = self.relu(self.batchnorm1(self.conv3b(x)))
        x = self.pool3(x)
        print_f("conv3 out shape: {}".format(x.shape))
        # x= self.batchnorm1(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        print_f("conv4 out shape: {}".format(x.shape))

        attn_x = self.relu(self.attn_conv4_temp(x))
        sig_attn = torch.sigmoid(attn_x)
        print_f("conv4 attn out shape: {}".format(attn_x.shape))
        x = torch.cat((x,x*sig_attn),2)
        x = self.relu(self.attn_conv4_temp_conv(x))
        # pdb.set_trace()

        attn_x = self.relu(self.attn_conv4_channel(x))
        sig_attn = torch.sigmoid(attn_x)
        print_f("conv4 attn out shape: {}".format(attn_x.shape))
        x = torch.cat((x,x*sig_attn),1)
        x = self.relu(self.attn_conv4_channel_conv(x))
        print_f("input after conv4 attn shape: {}".format(x.shape))

        # attn_x = self.relu(self.attn_conv4(x))
        # sig_attn = torch.sigmoid(attn_x)
        # x = x*sig_attn
        # print_f("conv4 attn out shape: {}".format(attn_x.shape))

        x = self.relu(self.conv5a(x))
        x = self.relu(self.batchnorm2(self.conv5b(x)))
        x = self.pool5(x)
        print_f("conv5 out shape: {}".format(x.shape))
        # x= self.batchnorm2(x)
        # print_f("conv_x_________",x)

        x = x.view(-1, 8192)
        x = self.relu(self.full_batchnorm1(self.fc6(x)))
        x = self.dropout(x)

        x = self.relu(self.full_batchnorm2(self.fc7(x)))
        x = self.dropout(x)

        # vid_mse_300 = self.fc8(x)
        logits = self.fc8(x)

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

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

if __name__ == "__main__":
    inputs = torch.rand(10, 3, 16, 112, 112)
    # att = torch.rand(50, 300)
    net = C3D(num_classes=101, pretrained=False)

    outputs= net.forward(inputs)
    print_f(outputs.size(), 1)
    # print_f(outputs2.size())
    # print_f(outputs3.size())
    # print_f(outputs4.size())
