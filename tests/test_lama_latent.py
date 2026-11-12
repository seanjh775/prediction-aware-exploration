"""Run with PYTHONPATH=lama python -m unittest discover -s tests."""
import unittest

import torch

from saicinpainting.training.modules.ffc import FFCResNetGenerator


class LatentGeneratorTest(unittest.TestCase):
    def make_generator(self, latent_dim=0):
        return FFCResNetGenerator(
            input_nc=4, output_nc=3, ngf=8, n_downsampling=1,
            n_blocks=1, max_features=16, add_out_act='sigmoid',
            init_conv_kwargs=dict(ratio_gin=0, ratio_gout=0),
            downsample_conv_kwargs=dict(ratio_gin=0, ratio_gout=0),
            resnet_conv_kwargs=dict(ratio_gin=0, ratio_gout=0),
            latent_dim=latent_dim).eval()

    def test_checkpoint_replay_diversity_and_gradients(self):
        torch.manual_seed(7)
        original = self.make_generator()
        model = self.make_generator(latent_dim=16)
        missing, unexpected = model.load_state_dict(original.state_dict(), strict=False)
        self.assertEqual(missing, ['latent_projection.weight'])
        self.assertEqual(unexpected, [])
        image = torch.rand(1, 3, 32, 32)
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 8:24, 8:24] = 1
        inputs = torch.cat([image * (1-mask), mask], dim=1)
        z = torch.randn(1, 16, requires_grad=True)
        baseline = original(inputs)
        self.assertTrue(torch.equal(baseline, model(inputs, torch.zeros_like(z))))
        first = model(inputs, z)
        self.assertTrue(torch.equal(first, model(inputs, z)))
        second = model(inputs, -z)
        self.assertGreater(((first-second).abs()*mask).sum().item(), 0)
        completion = first*mask + image*(1-mask)
        self.assertTrue(torch.equal(completion*(1-mask), image*(1-mask)))
        (first*mask).sum().backward()
        self.assertGreater(z.grad.abs().sum().item(), 0)
        self.assertGreater(model.latent_projection.weight.grad.abs().sum().item(), 0)
        restored = self.make_generator(latent_dim=16)
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(first, restored(inputs, z)))
        self.assertFalse(torch.equal(model(inputs), model(inputs)))

    def test_invalid_latent(self):
        inputs = torch.zeros(1, 4, 32, 32)
        with self.assertRaises(ValueError):
            self.make_generator()(inputs, torch.zeros(1, 16))
        with self.assertRaises(ValueError):
            self.make_generator(16)(inputs, torch.zeros(2, 16))


if __name__ == '__main__':
    unittest.main()
