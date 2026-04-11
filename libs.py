# libs

import numpy as np
# MPI replaced with stub for single-process / multiprocessing mode
class _MPIStub:
    COMM_WORLD = type('Comm', (), {
        'Get_rank': lambda self: 0,
        'Get_size': lambda self: 1,
        'bcast': lambda self, data, root=0: data,
        'allreduce': lambda self, val, op=None: val,
        'Scatterv': lambda self, *a, **kw: None,
    })()
    SUM = None
    DOUBLE = None
    @staticmethod
    def Get_library_version(): return "stub"
MPI = _MPIStub()
import psutil
import h5py
from scipy.interpolate import RegularGridInterpolator
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patches
import matplotlib.cm as cm
import time
from matplotlib.widgets import Button
import multiprocessing as mp
from functools import partial
import matplotlib.colors as mcolors
from matplotlib import colors
from matplotlib.lines import Line2D  # Import Line2D for the dummy legend entry
from tqdm import tqdm  # Import tqdm for progress reporting
import threading
import random
import sys
from matplotlib.patches import Arc, Wedge
from matplotlib.ticker import LogFormatter
import logging
import datetime
import os
import math
import shutil
import subprocess
from scipy.spatial import cKDTree
import json
