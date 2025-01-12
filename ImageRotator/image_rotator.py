# image_rotator.py

from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_file', help='input file path')
parser.add_argument('output_file', help='output file path')
parser.add_argument('angle', help='counterclockwise rotation (degrees)', type=int)
parser.add_argument('-i', '--info', action='store_true', help='display image size')

# parse
args = parser.parse_args()

# load
im = Image.open(args.input_file)

# display image size only if info flag is set to True
if args.info:
    print('input image dimensions:', im.size)

# rotate
im.rotate(args.angle)

# save
im.save(args.output_file)
