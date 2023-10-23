import argparse
import json
import multiprocessing
import subprocess
from taskqueue import LocalTaskQueue, RegisteredTask


cmd = "deepem/test/run.py"


class SingleRunTask(RegisteredTask):
    def __init__(self, opts, args):
        super(SingleRunTask, self).__init__(opts, args)
        self.opts = opts
        self.args = args

    def execute(self):
        if self.opts.dry_run:
            print(["python", cmd, self.args])
        else:
            subprocess.run(["python", cmd] + self.args.split())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--spec',
        type=str,
        required=True,
        help='local path to the JSON batch spec')
    parser.add_argument(
        '--iter', 
        type=int, 
        default=0)
    parser.add_argument(
        '--gpu_ids', 
        type=str, 
        default=['0'], 
        nargs='+')
    parser.add_argument(
        '--key_val',
        type=json.loads,
        default=None,
        help='extra key-value pairs to substitute')
    parser.add_argument(
        '--params', 
        type=str,
        required=True,
        help='inference parameters')
    parser.add_argument(
        '--dry_run',
        action='store_true')
    opts = parser.parse_args()

    # JSON batch spec
    with open(opts.spec, "r") as f:
        batch = json.load(f)

    # Run inference
    p = len(opts.gpu_ids)
    tasks = (SingleRunTask(opts, opts.params.format(**b, **opts.key_val, iter=opts.iter, gpu_id=opts.gpu_ids[i % p])) for i, b in enumerate(batch))
    with LocalTaskQueue(parallel=p) as tq:
        tq.insert_all(tasks)

    # for i, b in enumerate(batch):
    #     print(f"Batch run {i+1}")
    #     args = opts.params.format(**b, iter=opts.iter)
    #     if opts.dry_run:
    #         print(["python", cmd, args])
    #     else:
    #         subprocess.run(["python", cmd] + args.split())
