
import torch
import warnings
from thop import profile

from BaseSeg.network.get_model import UNet, EfficientSegNet

warnings.filterwarnings("ignore")


def get_layer_param(net):

    return sum([torch.numel(param) for param in net.parameters()])


def get_params_flops(net, input_array):
    flops, params = profile(net, inputs=(input_array))

    return params, flops


if __name__ == '__main__':

    model_cfg = {'NUM_CLASSES': 4,
                 'NUM_CHANNELS': [16, 32, 64, 128, 256],
                 'NUM_DEPTH': 4,
                 'NUM_BLOCKS': [1, 1, 1, 1],
                 # 'NUM_BLOCKS': [2, 2, 2, 2],
                 'DECODER_NUM_BLOCK': 1,
                 'AUXILIARY_TASK': False,
                 'AUXILIARY_CLASS': 1,
                 # 'ENCODER_CONV_BLOCK': 'ResECATwoLayerConvBlock',
                 # 'DECODER_CONV_BLOCK': 'ResECATwoLayerConvBlock',
                 'ENCODER_CONV_BLOCK': 'ResBaseConvBlock',
                 'DECODER_CONV_BLOCK': 'AnisotropicConvBlock',
                 'CONTEXT_BLOCK': None,
                 # 'CONTEXT_BLOCK': 'AnisotropicAvgPooling',
                 'INPUT_SIZE': [192, 192, 192],
                 # 'INPUT_SIZE': [160, 160, 160],
                 'WINDOW_LEVEL': [-325, 325],
                 'IS_PREPROCESS': False,
                 'IS_POSTPROCESS': False,
                 'IS_DYNAMIC_EMPTY_CACHE': False}

    model = EfficientSegNet(model_cfg).cuda()
    input_image = torch.randn([1, 1, 192, 192, 192]).float().cuda()
    #
    # input_image = torch.randn([1, 1, 160, 160, 160]).float().cuda()
    input_image = input_image.type(torch.cuda.FloatTensor)

    flops, params = profile(model, inputs=(input_image,))
    print("The flops is {} GB".format(flops / (1024 * 1024 * 1024)))
    print("The params is {} MB".format(params / (1024 * 1024)))



