#!/usr/bin/env python
# -*- coding: utf-8 -*-
from deepem.test.option import Options
from deepem.utils.onnx_utils import export_onnx


if __name__ == "__main__":
    # Options
    opt = Options().parse()
    export_onnx(opt, opt.chkpt_num)
