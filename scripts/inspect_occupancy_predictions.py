"""Render held-out predictions and attribute probability/geometry changes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt, uniform_filter

from models.probabilistic_lama import load_pretrained_big_lama,load_occupancy_checkpoint
from models.occupancy_data import OccupancyDataset,collate_occupancy
from scripts.train_probabilistic_lama import select_sample_ids


def error_counts(probability, target, eligible):
    predicted=probability>=.5; truth=target.astype(bool)
    return dict(tp=int((predicted & truth & eligible).sum()),
                fp=int((predicted & ~truth & eligible).sum()),
                fn=int((~predicted & truth & eligible).sum()),
                tn=int((~predicted & ~truth & eligible).sum()))


def transition_masks(p0,p1,target,eligible):
    before,after,truth=p0>=.5,p1>=.5,target.astype(bool)
    return dict(walls_correct_both=eligible & truth & before & after,
                walls_missed_both=eligible & truth & ~before & ~after,
                newly_missed_walls=eligible & truth & before & ~after,
                recovered_walls=eligible & truth & ~before & after,
                free_correct_both=eligible & ~truth & ~before & ~after,
                false_walls_both=eligible & ~truth & before & after,
                removed_false_walls=eligible & ~truth & before & ~after,
                new_false_walls=eligible & ~truth & ~before & after)


def focus_window(signal,size=64):
    height,width=signal.shape
    h,w=min(size,height),min(size,width)
    row,col=np.unravel_index(np.argmax(uniform_filter(signal.astype(float),size=(h,w),mode='constant')),signal.shape)
    return centered_window(row,col,signal.shape,size)


def centered_window(row,col,shape,size=64):
    height,width=shape;h,w=min(size,height),min(size,width)
    top,left=int(np.clip(row-h//2,0,height-h)),int(np.clip(col-w//2,0,width-w))
    return [top,left,top+h,left+w]


def error_image(probability,target,eligible):
    image=np.ones(target.shape+(3,))
    predicted=probability>=.5;truth=target.astype(bool)
    image[~eligible]=[.94,.90,.81]
    image[eligible & predicted & truth]=[.15,.15,.15]
    image[eligible & ~predicted & truth]=[.88,.18,.15]
    image[eligible & predicted & ~truth]=[.12,.40,.88]
    return image


def render(directory, sample, observation, target, eligible, probabilities, bounds, full_target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Patch
    directory.mkdir(parents=True,exist_ok=True)
    stages=['baseline','update50','update100'];labels=['Before training','Update 50','Update 100']
    unknown=np.isclose(observation,.5)
    figure,axes=plt.subplots(4,3,figsize=(14,17))
    axes[0,0].imshow(observation,cmap='Greys',vmin=0,vmax=1);axes[0,0].set_title('Observed input (gray = hidden)')
    axes[0,1].imshow(target,cmap='Greys',vmin=0,vmax=1);axes[0,1].set_title('Ground-truth KTH map')
    axes[0,2].imshow(full_target,cmap='Greys',vmin=0,vmax=1)
    origin=sample['crop_origin']
    axes[0,2].add_patch(Rectangle((origin[1],origin[0]),target.shape[1],target.shape[0],fill=False,color='#ef7f18',linewidth=2))
    axes[0,2].set_title('Crop location in complete floor map')
    for col,(stage,label_text) in enumerate(zip(stages,labels)):
        probability=probabilities[stage]
        complete=np.where(unknown,probability,target)
        heat=axes[1,col].imshow(complete,cmap='viridis',vmin=0,vmax=1)
        axes[1,col].set_title(label_text+' | P(occupied)')
        axes[2,col].imshow(complete>=.5,cmap='Greys',vmin=0,vmax=1)
        axes[2,col].set_title(label_text+' | 50% threshold')
        axes[3,col].imshow(error_image(probability,target,eligible))
        counts=sample['counts'][stage]
        axes[3,col].set_title('{} | missed {} / false {}'.format(label_text,counts['fn'],counts['fp']),fontsize=10)
    box=bounds['newly_missed_walls']
    for ax in [axes[0,1]]+list(axes[1:].ravel()):
        ax.add_patch(Rectangle((box[1]-.5,box[0]-.5),box[3]-box[1],box[2]-box[0],fill=False,color='#ef7f18',linewidth=1.5))
    for ax in axes.ravel():ax.axis('off')
    figure.legend(handles=[Patch(color=(.88,.18,.15),label='Missed real wall'),
                           Patch(color=(.12,.40,.88),label='False wall'),
                           Patch(color=(.15,.15,.15),label='Correct wall'),
                           Patch(color=(.94,.90,.81),label='Observed / excluded')],loc='lower center',ncol=4)
    figure.suptitle(sample['sample_id']+' | held-out validation | orange box = missed-wall zoom',fontsize=14)
    figure.subplots_adjust(top=.95,bottom=.05,left=.03,right=.91,hspace=.25,wspace=.08)
    cax=figure.add_axes([.93,.53,.012,.18])
    figure.colorbar(heat,cax=cax,label='0 = free, 1 = occupied')
    name=sample['sample_id']+'_overview.png';figure.savefig(str(directory/name),dpi=140);plt.close(figure)
    files=[name]
    for reason,box in bounds.items():
        top,left,bottom,right=box;crop=np.s_[top:bottom,left:right]
        fig,ax=plt.subplots(2,4,figsize=(15,8))
        ax[0,0].imshow(target[crop],cmap='Greys',vmin=0,vmax=1);ax[0,0].set_title('Ground truth (black = wall)')
        ax[1,0].imshow(observation[crop],cmap='Greys',vmin=0,vmax=1);ax[1,0].set_title('Observed input (gray = hidden)')
        for col,(stage,label_text) in enumerate(zip(stages,labels),1):
            complete=np.where(unknown,probabilities[stage],target)
            ax[0,col].imshow(complete[crop]>=.5,cmap='Greys',vmin=0,vmax=1);ax[0,col].set_title(label_text+' | binary')
            heat=ax[1,col].imshow(complete[crop],cmap='viridis',vmin=0,vmax=1);ax[1,col].set_title(label_text+' | probability')
        letter='B' if reason=='newly_missed_walls' else 'A'
        cell=sample.get('representative_cells',{}).get(letter)
        if cell is not None:
            x,y=cell['crop_col']-left,cell['crop_row']-top
            if 0<=x<right-left and 0<=y<bottom-top:
                for panel in ax.ravel():
                    panel.plot(x,y,marker='o',markersize=11,markerfacecolor='none',markeredgecolor='#ef7f18',markeredgewidth=2)
                    panel.text(min(right-left-6,x+2),max(3,y-2),letter,color='#ef7f18',weight='bold',fontsize=12)
        for panel in ax.ravel():panel.axis('off')
        fig.subplots_adjust(top=.88,bottom=.05,left=.02,right=.93,hspace=.08,wspace=.05)
        fig.colorbar(heat,ax=list(ax[1]),fraction=.015,pad=.02,label='P(occupied)')
        title='A real wall crosses below the cutoff (B)' if reason=='newly_missed_walls' else 'Lower penalty, but still a missed wall (A)'
        fig.suptitle('{}\n{} | crop rows {}:{}, columns {}:{}'.format(title,sample['sample_id'],top,bottom,left,right),fontsize=13)
        name=sample['sample_id']+'_'+reason+'.png';fig.savefig(str(directory/name),dpi=150);plt.close(fig);files.append(name)
    return files


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('work/probabilistic_two_fixed_100'))
    parser.add_argument('--output',type=Path,default=Path('work/prediction_inspection'))
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Choose a fresh output directory')
    args.output.mkdir(parents=True)
    config=json.loads((args.run/'config.json').read_text())
    split=json.loads((Path(config['data'])/'map_split.json').read_text())
    assert [len(split[k]) for k in ['train','validation','test']]==[10,2,2]
    data=OccupancyDataset(config['data'],'validation',config['validation_crop_size'],config['seed']+1,fixed_crops=True)
    select_sample_ids(data,config['validation_sample_ids'])
    batches=[collate_occupancy([data[i]]) for i in range(len(data))]
    logits={}
    for stage,path in [('baseline',None),('update50',args.run/'step_0050.pt'),('update100',args.run/'best.pt')]:
        model=load_pretrained_big_lama(config['pretrained']).eval() if path is None else load_occupancy_checkpoint(path)
        logits[stage]=[model(batch['observation'],batch['unknown_mask'])[0,0].numpy() for batch in batches]
        del model
    groups_total={};samples=[]
    for index,batch in enumerate(batches):
        target=batch['ground_truth'][0,0].numpy();observation=batch['observation'][0,0].numpy()
        eligible=(batch['unknown_mask']*batch['image_mask'])[0,0].numpy().astype(bool)
        probability={stage:torch.from_numpy(values[index]).sigmoid().numpy() for stage,values in logits.items()}
        losses={stage:F.binary_cross_entropy_with_logits(torch.from_numpy(values[index]),torch.from_numpy(target),reduction='none').numpy().astype(float)
                for stage,values in logits.items()}
        masks=transition_masks(probability['baseline'],probability['update100'],target,eligible)
        groups={}
        for name,mask in masks.items():
            count=int(mask.sum())
            groups[name]=dict(cells=count,nll_improvement_sum=float((losses['baseline']-losses['update100'])[mask].sum()),
                             mean_probability={s:float(p[mask].mean()) if count else None for s,p in probability.items()},
                             mean_nll={s:float(l[mask].mean()) if count else None for s,l in losses.items()})
            total=groups_total.setdefault(name,dict(cells=0,nll_improvement_sum=0.))
            total['cells']+=count;total['nll_improvement_sum']+=groups[name]['nll_improvement_sum']
        prediction=(probability['update100']>=.5)&eligible
        missed=(target==1)&~prediction&eligible
        distance=distance_transform_edt(~prediction) if prediction.any() else np.full(target.shape,np.inf)
        lost=masks['newly_missed_walls']
        bounds=dict(newly_missed_walls=focus_window(lost),
                    probability_improvement=focus_window(np.maximum(losses['baseline']-losses['update100'],0)*eligible))
        representatives={}
        candidate=masks['walls_missed_both'] & (probability['update100']>=.1) & (losses['baseline']>losses['update100'])
        if not candidate.any():candidate=masks['walls_missed_both'] & (losses['baseline']>losses['update100'])
        candidates=dict(A=candidate,B=lost)
        for letter,mask in candidates.items():
            if not mask.any():continue
            score=(losses['baseline']-losses['update100']) if letter=='A' else (probability['baseline']-probability['update100'])
            row,col=np.unravel_index(np.argmax(np.where(mask,score,-np.inf)),target.shape)
            representatives[letter]=dict(crop_row=int(row),crop_col=int(col),occupied_target=int(target[row,col]),
                probabilities={stage:float(torch.tensor(logits[stage][index][row,col],dtype=torch.float64).sigmoid()) for stage in logits},
                nll={stage:float(value[row,col]) for stage,value in losses.items()})
            reason='probability_improvement' if letter=='A' else 'newly_missed_walls'
            bounds[reason]=centered_window(row,col,target.shape)
        sample=dict(sample_id=batch['sample_id'][0],map_id=batch['map_id'][0],crop_origin=list(batch['crop_origin'][0]),
                    counts={s:error_counts(p,target,eligible) for s,p in probability.items()},transitions=groups,
                    missed_wall_distance=dict(total=int(missed.sum()),within_one_pixel=int((missed&(distance<=1)).sum()),
                        farther_than_one_pixel=int((missed&(distance>1)).sum())),
                    newly_missed_wall_distance=dict(total=int(lost.sum()),within_one_pixel=int((lost&(distance<=1)).sum()),
                        farther_than_one_pixel=int((lost&(distance>1)).sum())),focus_windows=bounds,
                    representative_cells=representatives)
        with np.load(str(Path(config['data'])/data.records[index]['file'])) as original:
            full_target=original['ground_truth']
        height,width=data[index]['ground_truth'].shape[-2:]
        # Trim computational padding before plotting. Scored masks already exclude it.
        sample['images']=render(args.output/'images',sample,observation[:height,:width],target[:height,:width],
                                eligible[:height,:width],{s:p[:height,:width] for s,p in probability.items()},bounds,full_target)
        np.savez_compressed(str(args.output/(sample['sample_id']+'.npz')),target=target,observation=observation,eligible=eligible,
                            **{stage:values[index] for stage,values in logits.items()})
        samples.append(sample)
    assert sum(g['cells'] for g in groups_total.values())==sum(sum(c['counts']['baseline'].values()) for c in samples)
    threshold_sweep=[]
    for threshold in [.05,.1,.2,.3,.4,.5,.6,.7,.8,.9]:
        tp=fp=fn=0
        for index,batch in enumerate(batches):
            eligible=(batch['unknown_mask']*batch['image_mask'])[0,0].numpy().astype(bool)
            target=batch['ground_truth'][0,0].numpy().astype(bool)
            predicted=torch.from_numpy(logits['update100'][index]).sigmoid().numpy()>=threshold
            tp+=int((predicted&target&eligible).sum());fp+=int((predicted&~target&eligible).sum());fn+=int((~predicted&target&eligible).sum())
        threshold_sweep.append(dict(threshold=threshold,occupied_iou=tp/(tp+fp+fn),tp=tp,fp=fp,fn=fn))
    report=dict(policy='Fixed 10/2/2 source-map split; validation-only inspection, no training or test data.',
                stages=['baseline','update50','update100'],samples=samples,aggregate_transitions=groups_total,
                update100_threshold_sweep=threshold_sweep,
                notes=['Ground truth is the KTH floor map, not a photograph of a real building.',
                       'Unknown cells only are counted; observed cells are composited exactly for display.',
                       'Zooms are selected to illustrate new missed-wall concentration and loss improvement, not randomly.',
                       'One-pixel proximity suggests a nearby prediction; it does not prove a rigid spatial shift.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(aggregate_transitions=groups_total,samples=[{k:v for k,v in s.items() if k not in ['transitions','images']} for s in samples]),indent=2),flush=True)


if __name__=='__main__':main()
