"""Fine-tune Big LaMa with masked Bernoulli NLL; no critic or latent sampling."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.probabilistic_lama import load_pretrained_big_lama
from models.occupancy_data import OccupancyDataset, collate_occupancy
from models.occupancy_objective import masked_bernoulli_loss, OccupancyMetrics


def select_training_maps(dataset, maps):
    if not maps:
        return
    available = {r['map_id'] for r in dataset.records}
    if not set(maps) <= available:
        raise ValueError('Requested maps must belong to the training split')
    dataset.records = [r for r in dataset.records if r['map_id'] in set(maps)]


def select_validation_examples(dataset, count):
    if count < 0:
        raise ValueError('Validation samples per map must be nonnegative')
    if not count:
        return
    selected = []
    for map_id in sorted({r['map_id'] for r in dataset.records}):
        records = sorted([r for r in dataset.records if r['map_id'] == map_id], key=lambda r:r['coverage'])
        indices = np.linspace(0, len(records)-1, min(count,len(records)), dtype=int)
        selected.extend(records[index] for index in indices)
    dataset.records = selected


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, limit=0):
    model.eval()
    total, by_map, by_coverage = OccupancyMetrics(), {}, {}
    for index, batch in enumerate(loader):
        if limit and index >= limit:
            break
        batch = to_device(batch, device)
        logits = model(batch['observation'], batch['unknown_mask'])
        args = (logits, batch['ground_truth'], batch['unknown_mask'], batch['image_mask'])
        total.update(*args)
        for item, map_id in enumerate(batch['map_id']):
            item_args = tuple(value[item:item+1] for value in args)
            by_map.setdefault(map_id, OccupancyMetrics()).update(*item_args)
            real = batch['image_mask'][item].sum().item()
            observed = 1-(batch['unknown_mask'][item]*batch['image_mask'][item]).sum().item()/real
            bucket = min(3, int(observed*4))
            label = '{}-{}% observed image cells'.format(bucket*25, (bucket+1)*25)
            by_coverage.setdefault(label, OccupancyMetrics()).update(*item_args)
    if not total.count:
        raise ValueError('Validation contains no eligible cells')
    return dict(global_metrics=total.compute(),
                by_map={key: value.compute() for key, value in by_map.items()},
                by_observed_coverage={key: value.compute() for key, value in by_coverage.items()})


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=root/'data'/'kth_masked_pilot_v1')
    parser.add_argument('--pretrained', type=Path, default=root/'pretrained_models'/'weights'/'big_lama')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--train-maps', nargs='+', help='Optional subset of original training-map IDs')
    parser.add_argument('--validation-samples-per-map', type=int, default=0,
                        help='0 uses all; otherwise take fixed coverage-spaced examples per held-out map')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--validate-every', type=int, default=100)
    parser.add_argument('--validation-limit', type=int, default=0, help='0 evaluates every validation example')
    parser.add_argument('--validation-crop-size', type=int, default=0,
                        help='0 uses full maps; nonzero is for inexpensive smoke checks')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.crop_size, args.validate_every) < 1 or args.lr <= 0:
        raise ValueError('Steps, batch size, crop size, validation interval, and learning rate must be positive')
    if args.output.exists():
        raise ValueError('Use a new output directory to preserve existing runs')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train = OccupancyDataset(args.data, 'train', args.crop_size, args.seed)
    validation = OccupancyDataset(args.data, 'validation', args.validation_crop_size, args.seed+1)
    select_training_maps(train, args.train_maps)
    select_validation_examples(validation, args.validation_samples_per_map)
    validation_loader = DataLoader(validation, batch_size=1, collate_fn=collate_occupancy)
    model = load_pretrained_big_lama(args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output.mkdir(parents=True)
    configuration = {key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()}
    configuration.update(objective='unweighted masked BCEWithLogits',
                         generator_config=model.generator_config, critic=False, latent_noise=False,
                         head_initialization='mean pretrained RGB occupancy logits', encoding_version=2,
                         training_examples=len(train), validation_examples=len(validation),
                         validation_sample_ids=[r['sample_id'] for r in validation.records])
    (args.output/'config.json').write_text(json.dumps(configuration, indent=2)+'\n')
    started = time.perf_counter()
    baseline = evaluate(model, validation_loader, device, args.validation_limit)
    (args.output/'baseline.json').write_text(json.dumps(baseline, indent=2)+'\n')
    print(json.dumps(dict(step=0, validation=baseline['global_metrics'],
                         baseline_seconds=time.perf_counter()-started)), flush=True)
    step, epoch, best = 0, 0, baseline['global_metrics']['nll']
    torch.save(dict(step=0, epoch=0, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    configuration=configuration, validation=baseline, torch_rng_state=torch.get_rng_state()),
               str(args.output/'best.pt'))
    with (args.output/'metrics.jsonl').open('w') as log:
        while step < args.steps:
            train.epoch = epoch
            order_rng = torch.Generator().manual_seed(args.seed+epoch)
            loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=order_rng,
                                collate_fn=collate_occupancy, num_workers=0)
            for batch in loader:
                model.train()
                batch = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                step_started = time.perf_counter()
                logits = model(batch['observation'], batch['unknown_mask'])
                loss = masked_bernoulli_loss(logits, batch['ground_truth'], batch['unknown_mask'], batch['image_mask'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError('Non-finite model gradient')
                optimizer.step(); step += 1
                record = dict(step=step, epoch=epoch, train_nll=loss.item(), sample_ids=batch['sample_id'],
                              train_seconds=time.perf_counter()-step_started)
                if step % args.validate_every == 0 or step == args.steps:
                    record['validation'] = evaluate(model, validation_loader, device, args.validation_limit)
                    checkpoint = dict(step=step, epoch=epoch, model=model.state_dict(),
                                      optimizer=optimizer.state_dict(), configuration=configuration,
                                      validation=record['validation'], torch_rng_state=torch.get_rng_state())
                    torch.save(checkpoint, str(args.output/'last.pt'))
                    nll = record['validation']['global_metrics']['nll']
                    if nll < best:
                        best = nll; torch.save(checkpoint, str(args.output/'best.pt'))
                log.write(json.dumps(record)+'\n'); log.flush()
                print(json.dumps(record if 'validation' not in record else dict(
                    step=step, train_nll=loss.item(), train_seconds=record['train_seconds'],
                    validation=record['validation']['global_metrics'])), flush=True)
                if step == args.steps:
                    break
            epoch += 1


if __name__ == '__main__':
    main()
