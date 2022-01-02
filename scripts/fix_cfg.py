from absl import flags, app
import wandb

from dlcomp.util import cleanup_wandb_config

FLAGS = flags.FLAGS

flags.DEFINE_string('sweep', None, 'sweep id', required=True)


def main(vargs):
    api = wandb.Api()
    sweep = api.sweep(f'sehoffmann/dlcomp/{FLAGS.sweep}')
    for run in sweep.runs:
        print(f'pruning {run.id}')
        run.config = cleanup_wandb_config(run.config, update=False)
        run.update()

if __name__ == '__main__':
    app.run(main)