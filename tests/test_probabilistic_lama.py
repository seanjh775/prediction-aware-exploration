import math
import unittest
import tempfile
from pathlib import Path

import torch

from models.probabilistic_lama import ProbabilisticLaMa, load_pretrained_big_lama, load_occupancy_checkpoint
from saicinpainting.training.modules.ffc import FFCResNetGenerator
from omegaconf import OmegaConf
from models.occupancy_data import collate_occupancy
from models.occupancy_objective import masked_bernoulli_loss, OccupancyMetrics


def tiny_model():
    return ProbabilisticLaMa(dict(input_nc=4, ngf=8, n_downsampling=1, n_blocks=1, max_features=16,
        init_conv_kwargs=dict(ratio_gin=0, ratio_gout=0),
        downsample_conv_kwargs=dict(ratio_gin=0, ratio_gout=0),
        resnet_conv_kwargs=dict(ratio_gin=0, ratio_gout=0)))


class ProbabilisticTest(unittest.TestCase):
    def test_masked_loss_and_gradient_exclude_known_and_padding(self):
        logits = torch.zeros(1,1,2,2, requires_grad=True)
        target = torch.tensor([[[[1.,0.],[1.,0.]]]])
        unknown = torch.tensor([[[[1.,0.],[1.,1.]]]])
        image = torch.tensor([[[[1.,1.],[0.,1.]]]])
        loss = masked_bernoulli_loss(logits, target, unknown, image)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertEqual(logits.grad[0,0,0,1].item(), 0)
        self.assertEqual(logits.grad[0,0,1,0].item(), 0)
        self.assertAlmostEqual(logits.grad[0,0,0,0].item(), -.25)
        with self.assertRaises(ValueError):
            masked_bernoulli_loss(logits, target, unknown*0, image)

    def test_probability_metrics_have_known_values(self):
        logits = torch.zeros(1,1,2,2)
        target = torch.tensor([[[[1.,0.],[1.,0.]]]])
        mask = torch.ones_like(logits)
        metric = OccupancyMetrics(); metric.update(logits, target, mask, mask)
        result = metric.compute()
        self.assertAlmostEqual(result['nll'], math.log(2), places=6)
        self.assertEqual(result['brier'], .25)
        self.assertEqual(result['ece'], 0)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['occupied_iou'], .5)

    def test_padding_alignment_and_loss_mask(self):
        def sample(h,w):
            return dict(observation=torch.full((1,h,w),.5), unknown_mask=torch.ones(1,h,w),
                        ground_truth=torch.zeros(1,h,w), map_id='a', sample_id='a')
        batch = collate_occupancy([sample(17,21), sample(20,18)])
        self.assertEqual(tuple(batch['observation'].shape), (2,1,32,32))
        self.assertEqual(batch['image_mask'].sum().item(), 17*21+20*18)
        self.assertEqual(batch['unknown_mask'].sum().item(), 17*21+20*18)
        self.assertEqual(batch['observation'][0,0,31,31].item(), 0)

    def test_single_pass_replay_shape_and_known_cell_preservation(self):
        model = tiny_model().eval()
        observation = torch.rand(1,1,33,41)
        mask = torch.zeros_like(observation); mask[:,:,8:24,8:24] = 1
        with torch.no_grad():
            first = model.predict(observation,mask); second = model.predict(observation,mask)
        self.assertEqual(tuple(first['logits'].shape), tuple(observation.shape))
        self.assertTrue(torch.equal(first['logits'],second['logits']))
        self.assertTrue(torch.equal(first['completed_probability']*(1-mask),observation*(1-mask)))
        self.assertEqual(first['variance'][mask==0].sum().item(),0)
        self.assertEqual(first['entropy'][mask==0].sum().item(),0)
        self.assertTrue(((first['occupancy_probability'] >= 0)&(first['occupancy_probability'] <= 1)).all())
        self.assertEqual(model.generator.latent_dim,0)

    def test_checkpoint_head_conversion_and_restore(self):
        config = dict(tiny_model().generator_config)
        config.update(output_nc=3, add_out_act='sigmoid')
        original = FFCResNetGenerator(**config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); (path/'models').mkdir()
            OmegaConf.save(OmegaConf.create(dict(generator=dict(config,kind='ffc_resnet'))), str(path/'config.yaml'))
            torch.save(dict(state_dict={'generator.'+k:v for k,v in original.state_dict().items()}), str(path/'models'/'best.ckpt'))
            model = load_pretrained_big_lama(path)
            original.eval(); model.eval()
            observation = torch.rand(1,1,32,32)
            unknown = torch.zeros_like(observation); unknown[:,:,8:24,8:24] = 1
            with torch.no_grad():
                original_rgb = original(torch.cat([(observation*(1-unknown)).repeat(1,3,1,1),unknown],1))
                expected_logits = torch.logit(original_rgb).mean(1,keepdim=True)
                self.assertTrue(torch.allclose(model(observation,unknown),expected_logits,atol=1e-5,rtol=1e-4))
            head_name = [name for name, layer in model.generator.named_modules()
                         if isinstance(layer,torch.nn.Conv2d)][-1]
            for key,value in model.generator.state_dict().items():
                expected = original.state_dict()[key]
                if key in [head_name+'.weight',head_name+'.bias']:
                    expected = expected.mean(0,keepdim=True)
                self.assertTrue(torch.equal(value,expected),key)
            torch.save(dict(configuration=dict(generator_config=model.generator_config,encoding_version=2),model=model.state_dict()),str(path/'trained.pt'))
            restored=load_occupancy_checkpoint(path/'trained.pt')
            self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items()))
            torch.save(dict(configuration=dict(generator_config=model.generator_config),model=model.state_dict()),str(path/'old.pt'))
            with self.assertRaisesRegex(ValueError,'corrected occupancy encoding'):
                load_occupancy_checkpoint(path/'old.pt')

    def test_input_encoding_and_hidden_label_invariance(self):
        model=tiny_model().eval()
        observed=torch.zeros(1,1,32,32); observed[:,:,0,0]=1
        mask=torch.zeros_like(observed);mask[:,:,8:24,8:24]=1
        captured=[]
        hook=model.generator.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
        with torch.no_grad():
            first=model(observed,mask)
            changed=observed.clone(); changed[mask.bool()]=.9
            second=model(changed,mask)
        hook.remove()
        self.assertTrue(torch.equal(first,second))
        self.assertEqual(captured[0][0,0,0,0].item(),1)  # occupied input is white
        self.assertEqual(captured[0][0,0,1,1].item(),0)  # free input is black
        self.assertEqual(captured[0][0,0,10,10].item(),0)  # hidden content removed


if __name__ == '__main__':
    unittest.main()
