"""Export the selected pilot examples to a GitHub-viewable PNG gallery."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil


def limited_selection(records, count, seed):
    import numpy as np
    if count < 1 or count > len(records):
        raise ValueError('Limit must be between 1 and the available review count')
    indices = np.random.RandomState(seed).choice(len(records), count, replace=False)
    return [records[index] for index in sorted(indices.tolist())]


def render_subset_pages(output, destination, records):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pages = []
    for start in range(0, len(records), 6):
        batch = records[start:start+6]
        figure, axes = plt.subplots(len(batch), 3, figsize=(15, 2.5*len(batch)), squeeze=False)
        for row, record in enumerate(batch):
            with np.load(str(output/record['file'])) as data:
                arrays = [1-data['ground_truth'], 1-data['observation'], data['unknown_mask']]
                titles = ['Ground truth', 'Observation | {:.1%} coverage'.format(record['coverage']),
                          'Mask: white = unknown']
                for column in range(3):
                    axes[row,column].imshow(arrays[column], cmap='gray', vmin=0, vmax=1)
                    axes[row,column].set_title(titles[column], fontsize=9)
                    axes[row,column].axis('off')
                axes[row,0].text(0, -.12, record['split']+' | '+record['sample_id'],
                                 transform=axes[row,0].transAxes, fontsize=8)
        figure.suptitle('Random pilot review subset | page {}'.format(len(pages)+1))
        figure.tight_layout(rect=(0,.01,1,.95), h_pad=2)
        page = 'contact_sheet_{:02d}.png'.format(len(pages)+1)
        figure.savefig(str(destination/page), dpi=130); plt.close(figure)
        pages.append(page)
    return pages


def export(output, destination, limit=None, subset_seed=20261113):
    manifest = json.loads((output/'manifest.json').read_text())
    selection = json.loads((output/'review'/'selection.json').read_text())
    parent_count = selection['selected']
    if limit is not None:
        selection = dict(selection)
        selection['records'] = limited_selection(selection['records'], limit, subset_seed)
        selection['selected'] = limit
        selection['parent_review_count'] = parent_count
        selection['parent_review_fraction'] = selection.pop('fraction')
        selection['published_fraction'] = limit / float(selection['total'])
        selection['subset_seed'] = subset_seed
    validation = json.loads((output/'validation_report.json').read_text())
    if validation['status'] != 'passed':
        raise ValueError('Validate the dataset before exporting its review')
    if destination.exists():
        raise ValueError('Choose a new review destination')
    destination.mkdir(parents=True)
    images = destination/'examples'
    images.mkdir()
    for record in selection['records']:
        for kind in ['mask','observation','ground_truth']:
            name = record['sample_id']+'_'+kind+'.png'
            shutil.copyfile(str(output/'review'/name), str(images/name))
    if limit is not None:
        selection['pages'] = render_subset_pages(output, destination, selection['records'])
    else:
        for page in selection['pages']:
            shutil.copyfile(str(output/'review'/page), str(destination/page))
    (destination/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    shutil.copyfile(str(output/'validation_report.json'), str(destination/'validation_report.json'))
    map_summary = [dict(map_id=m['map_id'], split=m['split'], samples=len(m['records']),
                        navigation_boundary=m.get('navigation_boundary', 'closed_occupancy_component'),
                        trajectories=m['trajectories'], source_sha256=m['source_sha256'])
                   for m in manifest['maps']]
    (destination/'map_summary.json').write_text(json.dumps(map_summary, indent=2)+'\n')
    reviewed_counts = Counter(r['split'] for r in selection['records'])
    description = ('{} examples generated across {} maps; {} examples published from the '
                   'seeded random review.'.format(len(manifest['records']), len(manifest['maps']), selection['selected']))
    if limit is not None:
        description += (' The full 20% review ({} examples) remains in the cloud workspace. '
                        'This gallery is a random subset of that review, sampled without replacement '
                        'using subset seed `{}` (parent selection seed `{}`).').format(
                            parent_count, subset_seed, selection['seed'])
    lines = ['# Masked pilot: random sample review', '', description, '',
             'The review is drawn from saved examples, not maps. All four trajectories per map '
             'use a 0.1 m synthetic circular footprint. Target coverage levels can be skipped '
             'when one scan crosses several levels or a trajectory exhausts its budget.', '',
             '| Split | Generated | In random review |', '| --- | ---: | ---: |']
    for split in ['train','validation','test']:
        lines.append('| {} | {} | {} |'.format(split, manifest['counts'][split], reviewed_counts[split]))
    lines += ['', 'Each contact sheet shows ground truth, partial observation, and unknown mask. '
              'Observation: black wall, white free, gray hidden. Mask: white hidden, black known. '
              'Test examples are included for dataset-integrity inspection only, not model tuning.', '',
              '## Contact sheets', '']
    for index, page in enumerate(selection['pages']):
        first, last = index*6+1, min((index+1)*6, selection['selected'])
        lines.append('- [Page {} — selected examples {}–{}]({})'.format(index+1, first, last, page))
    bounded_maps = [m['map_id'] for m in map_summary
                    if m['navigation_boundary'] == 'original_map_extent_virtual_boundary']
    if bounded_maps:
        lines += ['', 'Navigation exception: `{}` uses the original image extent as a virtual '
                  'boundary because its cropped floor opens into simulation padding. '
                  'Other maps use a closed occupancy component. See the dataset documentation '
                  'and trajectory details for this limitation.'.format('`, `'.join(bounded_maps))]
    lines += ['', '## Map collection results', '',
              '| Map | Split | Saved examples | Final trajectory coverage |', '| --- | --- | ---: | --- |']
    for item in map_summary:
        coverage = ', '.join('{:.1%}'.format(t['final_coverage']) for t in item['trajectories'])
        lines.append('| {} | {} | {} | {} |'.format(item['map_id'], item['split'], item['samples'], coverage))
    lines += ['', '## Individual PNGs', '', '| Sample | Split | Coverage | Target | Observation | Mask |',
              '| --- | --- | ---: | --- | --- | --- |']
    for record in selection['records']:
        stem = record['sample_id']
        links = ['[{}](examples/{}_{})'.format(label, stem, filename)
                 for label, filename in [('PNG','ground_truth.png'),('PNG','observation.png'),('PNG','mask.png')]]
        lines.append('| {} | {} | {:.1%} | {} |'.format(stem, record['split'], record['coverage'], ' | '.join(links)))
    lines += ['', '[Selection manifest](selection.json) · [Validation report](validation_report.json) '
              '· [Trajectory details](map_summary.json)', '',
              'Complete NPZ data is stored in the cloud workspace under `data/kth_masked_pilot_v1/` '
              'and excluded from Git. This gallery contains the selected PNGs only. '
              'See [dataset documentation](../masked_pilot.md) for generation and validation commands.', '']
    (destination/'README.md').write_text('\n'.join(lines))
    print('Exported {} review examples in {} contact sheets to {}'.format(
        selection['selected'], len(selection['pages']), destination))


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--destination', type=Path, default=root/'docs'/'pilot_review')
    parser.add_argument('--limit', type=int, help='Publish only this many randomly selected review examples')
    parser.add_argument('--subset-seed', type=int, default=20261113)
    arguments = parser.parse_args()
    export(arguments.output, arguments.destination, arguments.limit, arguments.subset_seed)
