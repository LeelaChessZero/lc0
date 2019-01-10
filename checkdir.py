#!/usr/bin/python

import sys
import os
if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
  exit(0)
exit(1)
