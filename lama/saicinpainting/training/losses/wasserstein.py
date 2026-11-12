"""Conditional WGAN-GP losses; minimize both returned objectives."""
import torch


def gradient_penalty(critic, real, fake, observation, unknown_mask):
    # Hold the condition fixed and interpolate only the candidate completion.
    # The critic composites known cells, so their candidate gradients are zero.
    alpha = torch.rand(real.shape[0], 1, 1, 1, device=real.device, dtype=real.dtype)
    interpolated = (alpha * real.detach() + (1-alpha) * fake.detach()).requires_grad_(True)
    scores = critic(interpolated, observation.detach(), unknown_mask.detach())
    gradients = torch.autograd.grad(
        outputs=scores, inputs=interpolated, grad_outputs=torch.ones_like(scores),
        create_graph=True, only_inputs=True)[0]
    norms = gradients.flatten(1).norm(2, dim=1)
    return ((norms - 1) ** 2).mean()


def critic_loss(critic, real, fake, observation, unknown_mask, gp_weight=10.0):
    if gp_weight < 0:
        raise ValueError('gp_weight must be nonnegative')
    if real.shape != fake.shape:
        raise ValueError('Real and fake maps must have matching shapes')
    if not torch.all(unknown_mask.flatten(1).sum(1) > 0):
        raise ValueError('Every training example must contain unknown cells')
    real, fake = real.detach(), fake.detach()
    observation, unknown_mask = observation.detach(), unknown_mask.detach()
    real_scores = critic(real, observation, unknown_mask)
    fake_scores = critic(fake, observation, unknown_mask)
    wasserstein = fake_scores.mean() - real_scores.mean()
    penalty = gradient_penalty(critic, real, fake, observation, unknown_mask)
    loss = wasserstein + gp_weight * penalty
    return loss, {
        'critic_loss': loss.detach(),
        'wasserstein_gap': (real_scores.mean() - fake_scores.mean()).detach(),
        'gradient_penalty': penalty.detach(),
        'real_score': real_scores.mean().detach(),
        'fake_score': fake_scores.mean().detach(),
    }


def generator_adversarial_loss(critic, fake, observation, unknown_mask):
    """Caller should freeze critic parameters during the generator update."""
    return -critic(fake, observation.detach(), unknown_mask.detach()).mean()
