import os
import gc
import math
import json
import copy
import torch
import random
import psutil
import logging
import argparse
import datasets
import threading
import transformers
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Dict
from dataclasses import dataclass

from torch.nn.utils import rnn
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from accelerate.logging import get_logger
from datasets import load_dataset, load_from_disk
from huggingface_hub import Repository, create_repo
from peft import (AutoPeftModelForCausalLM, LoraConfig, TaskType, get_peft_model)

from transformers import (
    get_scheduler,
    AutoTokenizer,
    SchedulerType,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
)

IGNORE_INDEX=-100

# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)

# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)

@dataclass
class DataCollatorForInstructionTuning:
    """Collate examples for instruction tuning."""

    tokenizer: PreTrainedTokenizerBase
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, labels = tuple([torch.tensor(feature[key]) for feature in features] for key in ["input_ids", "attention_mask", "labels"])
        input_ids=rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask=rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels=rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )