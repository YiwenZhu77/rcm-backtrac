# libs

import numpy as np
from mpi4py import MPI
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
