import os
import sys
import datetime
from collections import OrderedDict

import torch
import torch.distributed as dist


class Logger(object):
    def __init__(self, opt):
        self.monitor = {'train': Logger.Monitor(), 'test': Logger.Monitor()}
        self.log_dir = opt.log_dir
        self.in_spec = dict(opt.in_spec)
        self.out_spec = dict(opt.out_spec)
        self.outputsz = opt.outputsz
        self.lr = opt.lr

        # Metric learning
        self.delta_d = opt.delta_d

        # Basic logging
        self.timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.blv_num_channels = opt.blv_num_channels

        if opt.parallel == "DDP" and dist.get_rank() > 0:
            return

        self.log_params(vars(opt))
        self.log_command()
        self.log_command_args()

        # Blood vessel

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def record(self, phase, loss, nmsk, **kwargs):
        monitor = self.monitor[phase]

        # Reduce to scalar values.
        to_scalar = lambda x: x.item() if torch.is_tensor(x) else x
        for k in sorted(loss):
            monitor.add_to('vals', k, to_scalar(loss[k]))
            monitor.add_to('norm', k, to_scalar(nmsk[k]))

        for k, v in kwargs.items():
            monitor.add_to('vals', k, v)
            monitor.add_to('norm', k, 1)

    def check(self, phase, iter_num):
        stats = self.monitor[phase].flush()
        self.display(phase, iter_num, stats)
        return stats

    def display(self, phase, iter_num, stats):
        disp = "[%s] Iter: %8d, " % (phase, iter_num)
        for k, v in stats.items():
            disp += "%s = %.3f, " % (k, v)
        disp += "(lr = %.6f). " % self.lr
        print(disp)

    class Monitor(object):
        def __init__(self):
            self.vals = OrderedDict()
            self.norm = OrderedDict()

        def add_to(self, name, k, v):
            assert(name in ['vals','norm'])
            d = getattr(self, name)
            if k in d:
                d[k] += v
            else:
                d[k] = v

        def flush(self):
            ret = OrderedDict()
            for k in self.vals:
                if self.norm[k] == 0:
                    ret[k] = 0
                else:
                    ret[k] = self.vals[k]/self.norm[k]
            self.vals = OrderedDict()
            self.norm = OrderedDict()
            return ret

    def log_params(self, params):
        fname = os.path.join(self.log_dir, f"{self.timestamp}_params.csv")
        with open(fname, "w+") as f:
            for k, v in params.items():
                f.write(f"{k}: {v}\n")

    def log_command(self):
        fname = os.path.join(self.log_dir, f"{self.timestamp}_command")
        command = " ".join(sys.argv)
        with open(fname, "w+") as f:
            f.write(command)

    def log_command_args(self):
        fname = os.path.join(self.log_dir, f"{self.timestamp}_args.txt")
        with open(fname, "w+") as f:
            for arg in sys.argv[1:]:
                f.write(arg + "\n")
