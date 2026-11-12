import unittest

import torch

from saicinpainting.training.modules.conditional_wgan import ConditionalMapCritic
from saicinpainting.training.losses.wasserstein import (
    critic_loss, gradient_penalty, generator_adversarial_loss)


class WGANTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.critic = ConditionalMapCritic(base_channels=8, num_layers=2)
        self.real = torch.rand(2, 3, 32, 32)
        self.fake = torch.rand_like(self.real).requires_grad_(True)
        self.obs = torch.rand_like(self.real)
        self.mask = torch.zeros(2, 1, 32, 32)
        self.mask[:, :, 8:24, 8:24] = 1

    def test_update_is_finite_and_detaches_generator(self):
        optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4, betas=(0., .9))
        before = self.critic.features[0].weight.detach().clone()
        loss, metrics = critic_loss(self.critic, self.real, self.fake, self.obs, self.mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(metrics), {'critic_loss', 'wasserstein_gap', 'gradient_penalty',
                                        'real_score', 'fake_score'})
        loss.backward()
        self.assertIsNone(self.fake.grad)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.critic.parameters()
                            if p.grad is not None))
        optimizer.step()
        self.assertFalse(torch.equal(before, self.critic.features[0].weight))

    def test_known_region_and_unknown_placeholder_invariance(self):
        original = self.critic(self.fake, self.obs, self.mask)
        self.assertEqual(original.shape, (2,))
        changed_fake = self.fake.detach() + (1-self.mask)*100
        changed_obs = self.obs + self.mask*100
        self.assertTrue(torch.equal(original, self.critic(changed_fake, changed_obs, self.mask)))

    def test_generator_gradient_only_in_unknown_region(self):
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        loss = generator_adversarial_loss(self.critic, self.fake, self.obs, self.mask)
        loss.backward()
        self.assertGreater((self.fake.grad*self.mask).abs().sum().item(), 0)
        self.assertEqual((self.fake.grad*(1-self.mask)).abs().sum().item(), 0)

    def test_penalty_matches_linear_critic(self):
        def linear(candidate, observation, mask):
            # Exactly unit gradient norm over unknown dimensions.
            count = mask.flatten(1).sum(1)*candidate.shape[1]
            return (candidate*mask).flatten(1).sum(1)/count.sqrt()
        penalty = gradient_penalty(linear, self.real, self.fake, self.obs, self.mask)
        self.assertLess(penalty.item(), 1e-10)

    def test_rejects_empty_and_nonbinary_masks(self):
        with self.assertRaises(ValueError):
            critic_loss(self.critic, self.real, self.fake, self.obs, self.mask*0)
        with self.assertRaises(ValueError):
            self.critic(self.fake, self.obs, self.mask*.5)

    def test_loss_reaches_lama_latent_projection(self):
        from saicinpainting.training.modules.ffc import FFCResNetGenerator
        generator = FFCResNetGenerator(
            input_nc=4, output_nc=3, ngf=8, n_downsampling=1,
            n_blocks=1, max_features=16, add_out_act='sigmoid', latent_dim=16,
            init_conv_kwargs=dict(ratio_gin=0, ratio_gout=0),
            downsample_conv_kwargs=dict(ratio_gin=0, ratio_gout=0),
            resnet_conv_kwargs=dict(ratio_gin=0, ratio_gout=0))
        inputs = torch.cat([self.obs*(1-self.mask), self.mask], dim=1)
        fake = generator(inputs, torch.randn(2, 16))
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        generator_adversarial_loss(self.critic, fake, self.obs, self.mask).backward()
        gradient = generator.latent_projection.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
