import jittor as jt
import jittor.nn as nn
from jittor import Module


def make_layers_from_size(sizes, isFinal=False):
    layers = []
    for size in sizes:
        layers += [nn.Conv2d(size[0], size[1], kernel_size=3, padding=1), nn.BatchNorm2d(size[1], momentum=0.1), nn.ReLU()]
    if isFinal:
        layers.pop()
        layers.pop()
    return nn.Sequential(*layers)


class LinkNet_BackBone_FuseNet(Module):
    def __init__(self, num_labels, depth_nc):
        super(LinkNet_BackBone_FuseNet, self).__init__()

        ##### RGB ENCODER ####
        self.CBR1_RGB_ENC = make_layers_from_size([[3, 64], [64, 64]])
        self.pool1 = jt.nn.Pool(2, return_indices=True)

        self.CBR2_RGB_ENC = make_layers_from_size([[64, 128], [128, 128]])
        self.pool2 = jt.nn.Pool(2, return_indices=True)

        self.CBR3_RGB_ENC = make_layers_from_size([[128, 256], [256, 256], [256, 256]])
        self.pool3 = jt.nn.Pool(2, return_indices=True)
        self.dropout3 = nn.Dropout(0.4)

        self.CBR4_RGB_ENC = make_layers_from_size([[256, 512], [512, 512], [512, 512]])
        self.pool4 = jt.nn.Pool(2, return_indices=True)
        self.dropout4 = nn.Dropout(0.4)

        self.CBR5_RGB_ENC = make_layers_from_size([[512, 512], [512, 512], [512, 512]])
        self.dropout5 = nn.Dropout(0.4)

        self.pool5 = jt.nn.Pool(2, return_indices=True)

        ##### 3D DEPTH/DHAC ENCODER  ####

        self.CBR1_DEPTH_ENC = make_layers_from_size([[depth_nc, 64], [64, 64]])
        self.pool1_d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.CBR2_DEPTH_ENC = make_layers_from_size([[64, 128], [128, 128]])
        self.pool2_d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.CBR3_DEPTH_ENC = make_layers_from_size([[128, 256], [256, 256], [256, 256]])
        self.pool3_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3_d = nn.Dropout(0.4)

        self.CBR4_DEPTH_ENC = make_layers_from_size([[256, 512], [512, 512], [512, 512]])
        self.pool4_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4_d = nn.Dropout(0.4)

        self.CBR5_DEPTH_ENC = make_layers_from_size([[512, 512], [512, 512], [512, 512]])

        ####  RGB DECODER  ####
        self.unpool5 = jt.nn.MaxUnpool2d(2)
        self.CBR5_RGB_DEC = make_layers_from_size([[512, 512], [512, 512], [512, 512]])
        self.dropout5_dec = nn.Dropout(0.4)

        self.unpool4 = jt.nn.MaxUnpool2d(2)
        self.CBR4_RGB_DEC = make_layers_from_size([[512, 512], [512, 512], [512, 256]])
        self.dropout4_dec = nn.Dropout(0.4)

        self.unpool3 = jt.nn.MaxUnpool2d(2)
        self.CBR3_RGB_DEC = make_layers_from_size([[256, 256], [256, 256], [256, 128]])
        self.dropout3_dec = nn.Dropout(0.4)

        self.unpool2 = jt.nn.MaxUnpool2d(2)
        self.CBR2_RGB_DEC = make_layers_from_size([[128, 128], [128, 64]])

        self.unpool1 = jt.nn.MaxUnpool2d(2)
        self.CBR1_RGB_DEC = make_layers_from_size([[64, 64], [64, num_labels]], isFinal=True)

    def execute(self, rgb_inputs, depth_inputs):

        ########  DEPTH ENCODER  ########
        ########  RGB ENCODER  ########
        # Stage 1
        x = self.CBR1_DEPTH_ENC(depth_inputs)
        y = self.CBR1_RGB_ENC(rgb_inputs)
        y = (y + x) * 0.5
        x = self.pool1_d(x)
        y, id1 = self.pool1(y)

        # Stage 2
        x = self.CBR2_DEPTH_ENC(x)
        y = self.CBR2_RGB_ENC(y)
        y = (y + x) * 0.5
        x = self.pool2_d(x)
        y, id2 = self.pool2(y)

        # Stage 3
        x = self.CBR3_DEPTH_ENC(x)
        y = self.CBR3_RGB_ENC(y)
        y = (y + x) * 0.5
        x = self.pool3_d(x)
        x = self.dropout3_d(x)
        y, id3 = self.pool3(y)
        y = self.dropout3(y)

        # Stage 4
        x = self.CBR4_DEPTH_ENC(x)
        y = self.CBR4_RGB_ENC(y)
        y = (y + x) * 0.5
        x = self.pool4_d(x)
        x = self.dropout4_d(x)
        y, id4 = self.pool4(y)
        y = self.dropout4(y)

        # Stage 5
        x = self.CBR5_DEPTH_ENC(x)
        y = self.CBR5_RGB_ENC(y)
        y = (y + x) * 0.5
        prev_size = y.shape
        y, id5 = self.pool5(y)
        y = self.dropout5(y)

        ########  DECODER  ########

        # Stage 5 dec
        y = self.unpool5(y, id5, output_size=prev_size)
        y = self.CBR5_RGB_DEC(y)
        y = self.dropout5_dec(y)

        # Stage 4 dec
        y = self.unpool4(y, id4)
        y = self.CBR4_RGB_DEC(y)
        y = self.dropout4_dec(y)

        # Stage 3 dec
        y = self.unpool3(y, id3)
        y = self.CBR3_RGB_DEC(y)
        y = self.dropout3_dec(y)

        # Stage 2 dec
        y = self.unpool2(y, id2)
        y = self.CBR2_RGB_DEC(y)

        # Stage 1 dec
        y = self.unpool1(y, id1)
        y = self.CBR1_RGB_DEC(y)

        return y
