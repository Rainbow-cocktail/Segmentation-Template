from .vgg import vgg_encoder
from .resnet import resnet_encoders
from .densenet import densenet_encoders
from .dpn import dpn_encoders
from .senet import senet_encoders
from .octave import octave_encoders
from .cam import cam_encoders
from .mobilenetv3 import mobilenet_encoders
from .jpu import JPU
from .mit import mit_encoders

encoders = {}
encoders.update(vgg_encoder)
encoders.update(resnet_encoders)
encoders.update(densenet_encoders)
encoders.update(dpn_encoders)
encoders.update(senet_encoders)
encoders.update(octave_encoders)
encoders.update(cam_encoders)
encoders.update(mobilenet_encoders)
encoders.update(mit_encoders)


def get_encoder(name, input_channel=3):
    enc = encoders[name]['encoder']  # 获取编码器
    params = dict()
    params.update(encoders[name]['params'])
    params.update({'input_channel': input_channel})  # 这行代码是为了适配高光谱数据
    print(f"创建 {name} 编码器, 输入通道: {params['input_channel']}")
    encoder = enc(**params)
    return encoder, encoders[name]['out_shapes']


def get_encoder_names():
    return list(encoders.keys())




