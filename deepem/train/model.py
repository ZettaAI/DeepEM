import os
from io import BytesIO

import torch
import torch.nn as nn
from cloudfiles import CloudFiles, paths, exceptions


class Model(nn.Module):
    """
    Model wrapper for training.
    """

    def __init__(self, model, criteria, opt):
        super(Model, self).__init__()
        self.model = model
        self.criteria = criteria
        self.in_spec = dict(opt.in_spec)
        self.out_spec = dict(opt.out_spec)
        self.pretrain = opt.pretrain is not None

    def forward(self, sample):
        # Forward pass
        input_dict = {k: sample[k] for k in sorted(self.in_spec)}
        if len(input_dict) == 1:
            [input_tensor] = input_dict.values()
            preds = self.model(input_tensor)
        else:
            preds = self.model(input_dict)

        # Loss evaluation
        try:
            losses, nmasks = self.eval_loss(preds, sample)
        except:
            import pdb; pdb.set_trace()
            raise
        return losses, nmasks, preds

    def eval_loss(self, preds, sample):
        losses, nmasks = dict(), dict()
        for k in self.out_spec:
            target = sample[k]
            mask = sample[k + '_mask']
            criterion = self.criteria[k]
            if k == 'embedding':
                splt_key = k + '_split'
                splt = sample[splt_key] if splt_key in sample else None
                loss, nmsk = criterion(preds[k], target, mask, splt=splt)
            else:
                loss, nmsk = criterion(preds[k], target, mask)
            # PyTorch 0.4.0-specific workaround
            losses[k] = loss.unsqueeze(0)
            nmasks[k] = nmsk.unsqueeze(0)
        return losses, nmasks

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def save(self, fpath):
        torch.save(self.model.state_dict(), fpath)

    def load(self, fpath):
        chkpt = load_chkpt(fpath)
        # Backward compatibility
        state_dict = chkpt['state_dict'] if 'state_dict' in chkpt else chkpt
        if self.pretrain:
            model_dict = self.model.state_dict()
            state_dict = {k:v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)
        else:
            self.model.load_state_dict(state_dict)


class AmpModel(Model):
    def __init__(self, *args):
        super(AmpModel, self).__init__(*args)

    def forward(self, sample):
        with torch.cuda.amp.autocast():
            return super().forward(sample)


def load_chkpt(fpath):
    """Reads a (potentially remote) file and returns a file-like object."""
    if is_remote_fpath(fpath):
        remote_dir, basename = os.path.split(fpath)
        content = CloudFiles(remote_dir).get(basename)

        with BytesIO(content) as f:
            return torch.load(f)

    else:
        assert os.path.exists(fpath), f"no file ({fpath}) found"

        return torch.load(fpath)


def is_remote_fpath(fpath):
    """Tests whether a file path points to a remote location.

    Relies on cloudfiles.paths.extract
    """
    try:
        paths.extract(fpath)
        return True

    except exceptions.UnsupportedProtocolError:
        return False
