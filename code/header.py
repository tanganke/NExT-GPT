import argparse
import datetime
import json
import logging
import math
import os
import random
import re
import time
import types
from collections import OrderedDict
from copy import deepcopy

import deepspeed
import ipdb
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from torch.cuda.amp import GradScaler, autocast
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.deepspeed import HfDeepSpeedConfig

import data

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
