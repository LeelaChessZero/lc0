#!/usr/bin/env python3

import sys
import os
if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
  sys.exit(0)
sys.exit(1)
