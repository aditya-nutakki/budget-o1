import os, json
import re


def write_json(data, filepath, mode = "w"):
    with open(filepath, mode) as f:
        json.dump(data, f)


def read_json(path):
    with open(path, 'r') as f:
      data = json.load(f)
    return data
