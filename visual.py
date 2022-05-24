import os

import torch
from PIL import Image
import numpy as np

from configs import paths_config, global_config
from utils.log_utils import save_w, get_image_from_w
from utils.models_utils import toogle_grad


def load_G(model_path):
    with open(model_path, 'rb') as f:
        G = torch.load(f).to(global_config.device).eval()
    G = G.float()
    toogle_grad(G, False)
    return G


if __name__ == '__main__':
    G = load_G('/home/user/PTI/checkpoints/model_experiment_mystyle_img_16520883136504257_{\'quality\'= 0.pt')
    w = torch.load('/home/user/PTI/embeddings/test_data/PTI/img_16520883136504257_{\'quality\'= 0/0.pt').to(global_config.device)
    img = get_image_from_w(w, G)
    img = Image.fromarray(img, mode='RGB')
    os.makedirs('/home/user/PTI/result/inversion', exist_ok=True)
    img.save('/home/user/PTI/result/inversion/img_16520883136504257_{\'quality\'= 0.jpg')
