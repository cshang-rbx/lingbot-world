import argparse
from PIL import Image

parser = argparse.ArgumentParser(description="Resize image to 720p or 480p")
parser.add_argument("image", help="Input image path")
parser.add_argument("-o", "--output", help="Output path (default: overwrite input)")
parser.add_argument("--480", dest="use_480", action="store_true", help="Resize to 480p (832x480) instead of 720p")
args = parser.parse_args()

size = (832, 480) if args.use_480 else (1280, 720)
output = args.output or args.image

img = Image.open(args.image).convert("RGB")
img = img.resize(size, Image.LANCZOS)
img.save(output)
print(f"Saved: {output} ({size[0]}x{size[1]})")
