import os
from PIL import Image
from array import *
from random import shuffle

# Load from and save to
Names = [['C://Users//jbin7_000//Desktop//ten', 'train'], ['C://Users//jbin7_000//Desktop//ten', 'test']]

for name in Names:

    data_image = array('B')
    data_label = array('B')

    FileList = []
    for dirname in os.listdir(name[0])[1:]:  # [1:] Excludes .DS_Store from Mac OS
        path = Names[0][0]
        for filename in os.listdir(path):
            if filename.endswith(".png"):
                path1 = path + "//" + filename
                FileList.append(path1)

    shuffle(FileList)  # Usefull for further segmenting the validation set

    for filename in FileList:

        label = filename.split('_')[2]
        label = label.split(".")[0]
        print(filename)
        Im = Image.open(filename).convert('L')
        #Im = Image.open(filename,'r')
        pixel = Im.load()

        width, height = Im.size
        print(width)
        print(height)

        for x in range(0, width):
            for y in range(0, height):

                """if int(pixel[y,x]) == 255:
                    data_image.append(0)
                else:
                    data_image.append(int(pixel[y,x]))
                    """
                data_image.append(255-int(pixel[y, x]))
        data_label.append(int(label))  # labels start (one unsigned byte each)

    hexval = "{0:#0{1}x}".format(len(FileList), 6)  # number of files in HEX

    # header for label array

    header = array('B')
    header.extend([0, 0, 8, 1, 0, 0])
    header.append(int('0x' + hexval[2:][:2], 16))
    header.append(int('0x' + hexval[2:][2:], 16))

    data_label = header + data_label

    # additional header for images array

    if max([width, height]) <= 256:
        header.extend([0, 0, 0, width, 0, 0, 0, height])
    else:
        raise ValueError('Image exceeds maximum size: 256x256 pixels');

    header[3] = 3  # Changing MSB for image data (0x00000803)

    data_image = header + data_image
    print(name[0])

    output_file = open(name[0] + '-images-idx3-ubyte', 'wb')
    data_image.tofile(output_file)
    output_file.close()

    output_file = open(name[0] + '-labels-idx1-ubyte', 'wb')
    data_label.tofile(output_file)
    output_file.close()

# gzip resulting files
"""
for name in Names:
    os.system('gzip ' + name[1] + '-images-idx3-ubyte')
    os.system('gzip ' + name[1] + '-labels-idx1-ubyte')
"""
