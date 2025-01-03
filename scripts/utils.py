import torch
from typing import List, Tuple


import re
import torch
import io
import json


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        return open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def jsonl_load(file_path, text_key="prompt"):
    """
    Reads a .jsonl file line by line, parses each line as JSON,
    and extracts the value of `text_key` from each JSON object.

    :param file_path: Path to the .jsonl file.
    :param text_key: The JSON key whose value you want to collect as a string.
    :return: A list of strings extracted from the .jsonl file.
    """
    strings_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                # Skip any empty lines
                continue
            data = json.loads(line)
            # Assumes each JSON object has a field named text_key
            strings_list.append(data[text_key])
    return strings_list
