"""Locate held-out NLL regressions and isolate BN-buffer effects without training."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import label

from models.probabilistic_lama import ProbabilisticLaMa
from models.occupancy_data import OccupancyDataset, collate_occupancy
from scripts.train_probabilistic_lama import select_validation_examples


def group_summary(loss_before, loss_after, mask):
    count = int(mask.sum())
    return dict(cells=count, nll10=float(loss_before[mask].mean()) if count else None,
                nll20=float(loss_after[mask].mean()) if count else None,
                loss_sum_change=float((loss_after-loss_before)[mask].sum()))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('work/probabilistic_short_02'))
    parser.add_argument('--output', type=Path, default=Path('work/loss_diagnosis'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new diagnostic output directory')
    args.output.mkdir(parents=True)
    configuration = json.loads((args.run/'config.json').read_text())
    split = json.loads((Path(configuration['data'])/'map_split.json').read_text())
    if [len(split[key]) for key in ['train','validation','test']] != [10,2,2]:
        raise ValueError('The agreed 10/2/2 source-map split must remain fixed')
    best = torch.load(str(args.run/'best.pt'),map_location='cpu')
    if best['step'] != 10 or best['configuration'].get('encoding_version') != 2:
        raise ValueError('Expected corrected update-10 checkpoint')
    model = ProbabilisticLaMa(best['configuration']['generator_config']).eval()
    state10 = best['model']; del best
    last = torch.load(str(args.run/'last.pt'),map_location='cpu')
    if last['step'] != 20 or last['configuration'].get('encoding_version') != 2:
        raise ValueError('Expected corrected update-20 checkpoint')
    state20 = last['model']; del last
    data = OccupancyDataset(configuration['data'],'validation',configuration['validation_crop_size'],configuration['seed']+1)
    select_validation_examples(data,configuration['validation_samples_per_map'])
    if [record['sample_id'] for record in data.records] != configuration['validation_sample_ids']:
        raise ValueError('Validation selection differs from the original experiment')
    batches = [collate_occupancy([data[index]]) for index in range(len(data))]
    outputs = {}
    bn_names = [name for name,module in model.named_modules() if isinstance(module,torch.nn.modules.batchnorm._BatchNorm)]
    buffers = [name+'.'+suffix for name in bn_names for suffix in ['running_mean','running_var','num_batches_tracked']]
    cases = [('weights10_stats10',state10,state10),('weights20_stats20',state20,state20),
             ('weights20_stats10',state20,state10),('weights10_stats20',state10,state20)]
    case_metrics = {}
    for name, weights, statistics in cases:
        combined = dict(weights)
        combined.update({key:statistics[key] for key in buffers})
        model.load_state_dict(combined,strict=True)
        predicted = [model(batch['observation'],batch['unknown_mask']).cpu().numpy()[0,0] for batch in batches]
        outputs[name] = predicted
        loss_sum, cells = 0., 0
        for z,batch in zip(predicted,batches):
            eligible = (batch['unknown_mask']*batch['image_mask'])[0,0].bool()
            losses = F.binary_cross_entropy_with_logits(torch.from_numpy(z),batch['ground_truth'][0,0],reduction='none')
            loss_sum += losses[eligible].double().sum().item(); cells += int(eligible.sum())
        case_metrics[name] = dict(nll=loss_sum/cells,cells=cells)
        print(name,case_metrics[name],flush=True)
    all_before, all_after, all_target, all_prediction10, all_prediction20 = [],[],[],[],[]
    samples = []
    for index,batch in enumerate(batches):
        z10,z20 = outputs['weights10_stats10'][index],outputs['weights20_stats20'][index]
        target = batch['ground_truth'][0,0].numpy()
        eligible = (batch['unknown_mask']*batch['image_mask'])[0,0].numpy().astype(bool)
        before = F.binary_cross_entropy_with_logits(torch.from_numpy(z10),torch.from_numpy(target),reduction='none').numpy().astype(np.float64)
        after = F.binary_cross_entropy_with_logits(torch.from_numpy(z20),torch.from_numpy(target),reduction='none').numpy().astype(np.float64)
        delta = after-before
        origin = batch['crop_origin'][0]
        top = np.argsort(np.where(eligible,delta,-np.inf).ravel())[::-1][:10]
        top_cells = []
        for flat in top:
            row,col = np.unravel_index(flat,target.shape)
            a,b = float(z10[row,col]),float(z20[row,col])
            top_cells.append(dict(crop_row=int(row),crop_col=int(col),
                processed_map_row=int(origin[0]+row),processed_map_col=int(origin[1]+col),
                occupied_target=int(target[row,col]),logit10=a,logit20=b,
                occupied_probability10=float(torch.sigmoid(torch.tensor(a,dtype=torch.float64))),
                occupied_probability20=float(torch.sigmoid(torch.tensor(b,dtype=torch.float64))),
                nll10=float(before[row,col]),nll20=float(after[row,col]),increase=float(delta[row,col])))
        wrong20 = (z20>=0) != target.astype(bool)
        regions,count = label(eligible & wrong20 & (delta>1))
        region_results = []
        if count:
            weights = np.bincount(regions.ravel(),weights=np.where(regions>0,delta,0).ravel())
            for component in np.argsort(weights[1:])[::-1][:3]+1:
                rr,cc=np.where(regions==component)
                region_results.append(dict(cells=len(rr),loss_sum_increase=float(weights[component]),
                    crop_bounds=[int(rr.min()),int(cc.min()),int(rr.max()+1),int(cc.max()+1)],
                    processed_map_bounds=[int(rr.min()+origin[0]),int(cc.min()+origin[1]),
                                          int(rr.max()+1+origin[0]),int(cc.max()+1+origin[1])]))
        sample = dict(sample_id=batch['sample_id'][0],map_id=batch['map_id'][0],crop_origin=list(origin),
                      **group_summary(before,after,eligible))
        sample.update(top_cells=top_cells,largest_worsened_error_regions=region_results)
        samples.append(sample)
        np.savez_compressed(str(args.output/(sample['sample_id']+'.npz')),observation=batch['observation'][0,0].numpy(),
                            target=target,eligible=eligible,logits10=z10,logits20=z20,nll10=before,nll20=after)
        all_before.append(before[eligible]);all_after.append(after[eligible]);all_target.append(target[eligible])
        all_prediction10.append(z10[eligible]>=0);all_prediction20.append(z20[eligible]>=0)
    before,after,target = map(np.concatenate,[all_before,all_after,all_target])
    p10,p20 = map(np.concatenate,[all_prediction10,all_prediction20])
    good10,good20 = p10==target.astype(bool),p20==target.astype(bool)
    groups = {name:group_summary(before,after,mask) for name,mask in [
        ('occupied_targets',target==1),('free_targets',target==0),
        ('wrong_at_both',~good10 & ~good20),('new_errors',good10 & ~good20),
        ('corrected_errors',~good10 & good20),('correct_at_both',good10 & good20)]}
    delta=after-before;positive=delta[delta>0].sum()
    concentration = {}
    for fraction in [.01,.05,.1]:
        count=int(math.ceil(len(delta)*fraction))
        increase=np.sort(delta)[-count:].sum()
        concentration[str(fraction)]=dict(cells=count,loss_sum_increase=float(increase),
                                         fraction_of_positive_increase=float(increase/positive),
                                         fraction_of_net_increase=float(increase/delta.sum()))
    bn_modules = {name:module for name,module in model.named_modules() if name in bn_names}
    report = dict(policy='Fixed 10 training / 2 validation / 2 test maps; diagnosis uses validation only.',
                  checkpoints=[10,20],validation_cases=case_metrics,samples=samples,groups=groups,
                  concentration=concentration,batchnorm=dict(layers=len(bn_names),
                    momenta=sorted({module.momentum for module in bn_modules.values()}),
                    epsilons=sorted({module.eps for module in bn_modules.values()}),
                    counter_differences=sorted({int(state20[name+'.num_batches_tracked']-state10[name+'.num_batches_tracked']) for name in bn_names})),
                  limitations=['No weights were optimized and no test examples were loaded.',
                    'Swapping BN running buffers is an evaluation diagnostic, not a trained replacement model.',
                    'Coordinates refer to processed occupancy arrays, with bounds end-exclusive.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['groups','concentration','batchnorm']},indent=2),flush=True)


if __name__ == '__main__':
    main()
