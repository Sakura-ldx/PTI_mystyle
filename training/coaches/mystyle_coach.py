import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from configs import paths_config, hyperparameters, global_config
from models.StyleCLIP.criteria import id_loss
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
from utils.models_utils import toogle_grad, load_old_G


def save_image(img, dir, name):
    img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
    img = Image.fromarray(img, mode='RGB')
    img.save(os.path.join(dir, f'{name}.jpg'))


class MystyleCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)
        self.id_loss = id_loss.IDLoss().to(global_config.device).eval()

    def train(self):

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True
        w_pivots = []
        images = []

        self.G.synthesis.eval()
        self.G.mapping.eval()
        z_pivot = np.random.RandomState(256).randn(1, 512)
        with torch.no_grad():
            w_pivot = self.G.mapping(torch.from_numpy(z_pivot).to(global_config.device), None)
            w_pivot = w_pivot.clone().detach()
            image_pivot = self.forward(w_pivot)
            image_pivot = image_pivot.clone().detach()
        torch.save(w_pivot, f'{w_path_dir}/{paths_config.pti_results_keyword}/0.pt')
        dir = f'./result/{global_config.run_name}/origin'
        os.makedirs(dir, exist_ok=True)
        save_image(image_pivot[0], dir, '0.jpg')

        for i in tqdm(range(hyperparameters.max_pti_steps)):
            self.G.synthesis.train()
            self.G.mapping.train()

            z_edits = np.random.RandomState(123).randn(5, 512)
            w_edits = self.G.mapping(torch.from_numpy(z_edits).to(global_config.device), None)
            w_edits = w_edits.clone().detach()
            w_distances = (w_edits - w_pivot) / torch.norm(w_edits - w_pivot, 2, dim=2, keepdim=True)

            w_finals = []
            for seed in range(5):
                a = np.random.RandomState(seed).randn()
                for w_distance in w_distances:
                    w_finals.append(w_pivot + a * w_distance)
            w_finals = torch.cat(w_finals, dim=0)

            real_images_batch = image_pivot.repeat([5 * w_distances.shape[0], 1, 1, 1])

            generated_images = self.forward(w_finals)
            loss = self.id_loss(generated_images, real_images_batch)
            print(loss.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 50 == 0:
                self.G.synthesis.eval()
                self.G.mapping.eval()
                with torch.no_grad:
                    generated_images = self.forward(w_finals)

                dir = f'./result/{global_config.run_name}/train/{i}'
                os.makedirs(dir, exist_ok=True)
                name = 0
                for image in generated_images:
                    save_image(image, dir, f'{name}.jpg')
                    name += 1

            global_config.training_step += 1
        if self.use_wandb:
            log_images_from_w(w_pivots, self.G, [image[0] for image in images])

        torch.save(self.G,
                   f'{global_config.run_name}/{paths_config.checkpoints_dir}/model_mystyle.pt')