import imagehash
import tifffile
import imagecodecs

from   tabulate import tabulate
from   matplotlib import pyplot as plt
from   PIL import Image

def normalized_hamming_distance_hex(hash1, hash2):
    hash1            = str(hash1)
    hash2            = str(hash2)

    int1             = int(hash1, 16)
    int2             = int(hash2, 16)

    xor              = int1 ^ int2
    hamming_distance = bin(xor).count('1')
    max_distance     = len(hash1) * 4

    return hamming_distance / max_distance

def phash_level(filepath, level=0, rotate_degrees=None, resize_percentage=None):
    with tifffile.TiffFile(filepath) as tif:
        page = tif.pages[level]
        arr = page.asarray()
    img = Image.fromarray(arr)

    if rotate_degrees is not None:
        img = img.rotate(rotate_degrees, expand=False, fillcolor=(255, 255, 255))
    
    if resize_percentage is not None:
        #new_size = (800, 600)  # Example: 800 pixels wide, 600 pixels high
        
        original_width, original_height = img.size
        new_width = int(original_width * (resize_percentage / 100))
        new_height = int(original_height * (resize_percentage / 100))
        new_size = (new_width, new_height)
        print(new_size)
        img = img.resize(new_size)
    #plt.imshow(img)
    #plt.show()    
    return imagehash.phash(img)

def calculate_hamming_distance(first_wsi, second_wsi, rotation=0, resizepercentage = 100):
    print(f"\n{'Comparison':40} {'Hash 1':<32} {'Hash 2':<32} {'Hamming Distance'}")
    print("-" * 120)
    
    first_hash  = phash_level(first_wsi,  level=1)
    second_hash = phash_level(second_wsi, level=1, rotate_degrees=rotation, resize_percentage = resizepercentage)

    table_data  = []

    comp_name = f"{first_wsi} - {second_wsi}"
    dist      = normalized_hamming_distance_hex(first_hash, second_hash)

    table_data.append([comp_name, str(first_hash), str(second_hash), f"{dist:.4f}"])

    headers = ['Comparison', 'Hash 1', 'Hash 2', 'Normalized Hamming Distance']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

