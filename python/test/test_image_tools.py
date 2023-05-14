#!/usr/bin/env python3
"""test file for the image_tools """

import os
from subprocess import getstatusoutput, getoutput

prg = './image_tools.py'

def test_exists():
    assert os.path.isfile(prg)
