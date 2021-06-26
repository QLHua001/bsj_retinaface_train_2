import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH

if __name__ == "__main__":
    checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        # print(k)
        # print("v.shape: ", v.shape)
        name = k[7:]  # remove module.
        new_state_dict[name] = v

    backbone = MobileNetV1()
    # for name, module in backbone.state_dict().items():
    #     print(name)
    #     # print(module)
    body = _utils.IntermediateLayerGetter(backbone, {'stage2': 1, 'stage3': 2})

    out = body(torch.randn(1, 3, 144, 256))
    print(type(out))
    print([(k, v.shape) for k, v in out.items()])
    input = out.values()
    print("type(out.values()): ", type(input))
    input = list(out.values())
    print(len(input))
    print("type: ", type(input[0]))
    print("type: ", type(input[1]))
    print("values[0]:", input[0].shape)
    print("values[1]:", input[1].shape)

