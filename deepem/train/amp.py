"""Mixed-precision training.

One object owns every precision decision, so the model wrapper, the training
loop, eval and checkpointing cannot disagree about it -- which is how the old
path ended up running fp16 when bf16 was asked for (``AmpModel`` opened its own
dtype-less ``torch.cuda.amp.autocast()``, which defaults to fp16 and overrode
run.py's bf16 context).

The recipe is the standard one for PyTorch >= 2.x:

* Weights and optimizer state stay fp32. Autocast casts op inputs only.
* Autocast covers the network forward only. Predictions are cast back to fp32
  before the loss, so sigmoids, margins and the mean-shift reductions run in
  fp32. Backward runs outside autocast.
* bf16 has fp32's exponent range and needs no loss scaling. fp16 does: its
  GradScaler state is checkpointed so a resumed run keeps its scale.
"""

from contextlib import nullcontext

import torch


MODES = {
    None: None,
    'fp32': None,
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
}


class Precision:
    def __init__(self, mode=None, device_type='cuda'):
        if mode not in MODES:
            raise ValueError(f"unsupported mixed precision mode: {mode!r}")
        self.mode = mode or 'fp32'
        self.dtype = MODES[mode]
        self.device_type = device_type
        self.scaler = (torch.amp.GradScaler(device_type)
                       if self.dtype is torch.float16 else None)

    @classmethod
    def from_opt(cls, opt, device_type='cuda'):
        return cls(getattr(opt, 'mixed_precision', None), device_type=device_type)

    @property
    def enabled(self):
        return self.dtype is not None

    def autocast(self, device_type=None):
        """Context for the network forward; a no-op in fp32."""
        if not self.enabled:
            return nullcontext()
        return torch.autocast(device_type or self.device_type, dtype=self.dtype)

    def backward_step(self, loss, optimizer):
        """Backward and optimizer step, with loss scaling in fp16.

        Returns False when the fp16 scaler skipped the step because of inf/nan
        gradients, True otherwise.
        """
        if self.scaler is None:
            loss.backward()
            optimizer.step()
            return True
        scale = self.scaler.get_scale()
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        # update() lowers the scale exactly when step() was skipped.
        return self.scaler.get_scale() >= scale

    @property
    def loss_scale(self):
        return self.scaler.get_scale() if self.scaler is not None else None

    def state_dict(self):
        return {'mode': self.mode,
                'scaler': self.scaler.state_dict() if self.scaler is not None else None}

    def load_state_dict(self, state):
        """Restore the fp16 scaler. States from another mode are ignored."""
        if not state or self.scaler is None or state.get('scaler') is None:
            return
        self.scaler.load_state_dict(state['scaler'])

    def describe(self):
        return (f"precision={self.mode} "
                f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32} "
                f"matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}")
