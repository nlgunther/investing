import os, sys, re, numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy as scp
import seaborn as sns, cvxpy as cp
import yfinance as yf
from importlib import reload
from typing import List, Dict, Tuple, Optional, Union
# type: ignore

to_list = lambda f: lambda *args, **kwargs: list(f(*args, **kwargs))
lmap,lfilter = [to_list(f) for f in (map, filter)]