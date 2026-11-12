"""Single-pass occupancy probabilities using the pretrained Big LaMa backbone."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from saicinpainting.training.modules.ffc import FFCResNetGenerator


class ProbabilisticLaMa(nn.Module):
    def __init__(self, generator_config):
        super().__init__()
        self.generator_config = dict(generator_config)
        self.generator_config.update(output_nc=1, add_out_act=False, latent_dim=0)
        if self.generator_config.get('input_nc') != 4:
            raise ValueError('Expected pretrained RGB image plus mask input')
        self.generator = FFCResNetGenerator(**self.generator_config)

    def forward(self, observation, unknown_mask):
        if observation.ndim != 4 or observation.shape[1] != 1 or observation.shape != unknown_mask.shape:
            raise ValueError('Expected matching B1HW observation and mask')
        height, width = observation.shape[-2:]
        # Checkpoint images encode white/free as 1; dataset occupancy encodes walls as 1.
        image = ((1-observation) * (1-unknown_mask)).repeat(1, 3, 1, 1)
        inputs = torch.cat([image, unknown_mask], dim=1)
        inputs = F.pad(inputs, (0, (-width) % 16, 0, (-height) % 16))
        return self.generator(inputs)[:, :, :height, :width]

    @torch.no_grad()
    def predict(self, observation, unknown_mask):
        if self.training:
            raise RuntimeError('Call model.eval() before probability inference')
        logits = self(observation, unknown_mask)
        probability = logits.sigmoid()
        # Uncertainty applies to unseen cells; known observations are exact in this dataset.
        return dict(logits=logits, occupancy_probability=probability,
                    completed_probability=probability*unknown_mask + observation*(1-unknown_mask),
                    variance=probability*(1-probability)*unknown_mask,
                    entropy=(F.softplus(logits)-logits*probability).clamp_min(0)*unknown_mask)


def load_pretrained_big_lama(model_directory):
    directory = Path(model_directory)
    config = OmegaConf.load(str(directory/'config.yaml'))
    generator_config = OmegaConf.to_container(config.generator, resolve=True)
    if generator_config.pop('kind') != 'ffc_resnet':
        raise ValueError('This adapter requires an FFC Big LaMa checkpoint')
    model = ProbabilisticLaMa(generator_config)
    checkpoint = torch.load(str(directory/'models'/'best.ckpt'), map_location='cpu')
    state = {key[len('generator.'):]: value for key, value in checkpoint['state_dict'].items()
             if key.startswith('generator.')}
    head_name, head = [(name, layer) for name, layer in model.generator.named_modules()
                       if isinstance(layer, nn.Conv2d)][-1]
    for suffix in ['weight', 'bias']:
        key = head_name+'.'+suffix
        if key not in state or state[key].shape[0] != 3:
            raise ValueError('Expected the original three-channel output head: '+key)
        # Mean RGB free-space logits become occupancy logits by negating them.
        state[key] = -state[key].mean(dim=0, keepdim=True)
    model.generator.load_state_dict(state, strict=True)
    return model


def load_occupancy_checkpoint(path, device='cpu'):
    checkpoint = torch.load(str(path), map_location='cpu')
    model = ProbabilisticLaMa(checkpoint['configuration']['generator_config'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval()
