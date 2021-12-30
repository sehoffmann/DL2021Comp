from absl import flags, app
import wandb
import numpy as np

from dlcomp.util import cleanup_wandb_config

FLAGS = flags.FLAGS

flags.DEFINE_string('sweep', None, 'sweep id')
flags.DEFINE_integer('keep', None, 'how many files to keep', lower_bound=1, required=True)
flags.DEFINE_integer('max_epoch', None, 'maximum epoch to scan for')
flags.DEFINE_bool('dry', False, 'dry-run')


def prune_run(run):
    # find files
    max_epoch = FLAGS.max_epoch if FLAGS.max_epoch else run.config['epochs']
    candidates = [f'models/epoch{i}.pth' for i in range(0, max_epoch, run.config['save_every'])]
    print('\nchecking: ', ', '.join(candidates))
    files = []
    for file in run.files(candidates):
        if file.size != 0:
            files.append(file)
    
    n_files = len(files)
    if n_files == FLAGS.keep:
        print('\nnothing to prune')
        return
    
    subranges = np.array_split(np.arange(n_files), FLAGS.keep)
    to_keep = [subr[len(subr) // 2] for subr in subranges]
    print('\nkeeping: ', ', '.join(files[i].name for i in to_keep), '\n')

    for i, file in enumerate(files):
        if i in to_keep:
            continue
        print(f'deleting {file.name}')
        if not FLAGS.dry:
            file.delete()


def main(vargs):
    api = wandb.Api()
    sweep = api.sweep(f'sehoffmann/dlcomp/{FLAGS.sweep}')
    for run in sweep.runs:
        print(f'pruning {run.id}')
        prune_run(run)
        print('=' * 100)


if __name__ == '__main__':
    app.run(main)