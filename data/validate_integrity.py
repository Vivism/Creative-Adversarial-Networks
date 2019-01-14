'''
Script to check whether images are corrupted. Without an argument, checks `data/wikiart/`. Otherwise checks 
the `data/<dataset-name>`
Usage: 
    python3 validate_integrity.py <dataset-name>
    
    <dataset-name> : `data/<dataset-name>`
'''
from src.utils import test_images
from glob import glob
import sys

try:
	dataset = str(sys.argv[1])
except IndexError:
	dataset = "wikiart"

path = "./{}/*/*.jpg".format(dataset)
found = glob(path)
count = len(found)

print("{} images found -- validating their integrity...".format(count))

test_images(found)

