from glob import glob
L = [i for i in glob('Banana_Republic/Mens/Apparel/Shirt/*')]
from imutils import paths
import argparse
import pickle
import vptree
import time
import cv2
import os
import numpy as np
from PIL import Image
import imagehash
import streamlit as st

st.title('VPTree - colorHash demo')
def plt_imshow(title, image):
    # convert the image frame BGR to RGB color space and display it
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image,caption=title)


def colorhash2(image, bit):
  resized = image# image.resize((200,200))
  original_hash = imagehash.colorhash(resized,binbits=bit)
  hash_as_str = str(original_hash)
  restored_hash = imagehash.hex_to_flathash(hash_as_str, hashsize=3)
  str_as_int = int(hash_as_str, base=16)
  
  return restored_hash,str_as_int


def hamming(a, b):
  return bin(int(a) ^ int(b)).count("1")

def convert_hash(h):
    # convert the hash to NumPy's 64-bit float and then back to
    # Python's built in int
    return int(np.array(h, dtype="int64"))

def train():
    args = {
        "images": "Banana_Republic/Mens/Apparel/Shirt/",
        "tree": "vptree.pickle",
        "hashes": "hashes.pickle"
    }

    # grab the paths to the input images and initialize the dictionary
    # of hashes
    imagePaths = L
    hashes = {}
    bitx=7
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # load the input image
        print("[INFO] processing image {}/{}".format(i + 1,
            len(imagePaths)))
        image = Image.open(imagePath)

        # compute the hash for the image and convert it
        h1,h = colorhash2(image,bitx)
        #h = convert_hash(h)

        # update the hashes dictionary
        l = hashes.get(h, [])
        l.append(imagePath)
        hashes[h] = l

    # build the VP-Tree
    print("[INFO] building VP-Tree...")
    points = list(hashes.keys())
    tree = vptree.VPTree(points, hamming)

    # serialize the VP-Tree to disk
    print("[INFO] serializing VP-Tree...")
    f = open(args["tree"], "wb")
    f.write(pickle.dumps(tree))
    f.close()

    # serialize the hashes to dictionary
    print("[INFO] serializing hashes...")
    f = open(args["hashes"], "wb")
    f.write(pickle.dumps(hashes))
    f.close()

def query():
    image = st.file_uploader('Sube una imagen',accept_multiple_files=False)
    if image is not None:
        bytes_data = image.getvalue()
        image = cv2.imdecode(np.asarray(bytearray(bytes_data), dtype=np.uint8), cv2.IMREAD_COLOR)
        b,g,r = cv2.split(image)
        image2 = cv2.merge([r,g,b])
        st.image(image2)
        args = {
            "tree": "vptree.pickle",
            "hashes": "hashes.pickle",
            "distance": 10
        }
        print("[INFO] loading VP-Tree and hashes...")
        tree = pickle.loads(open(args["tree"], "rb").read())
        hashes = pickle.loads(open(args["hashes"], "rb").read())

        # compute the hash for the query image, then convert it
        x,queryHash = colorhash2(Image.fromarray(image2),7)
        #queryHash = convert_hash(queryHash)

        # load the input query image
        

        # perform the search
        st.subheader('Resultados de busqueda')
        print("[INFO] performing search...")
        start = time.time()
        results = tree.get_all_in_range(queryHash, args["distance"])
        #results = sorted(results)
        end = time.time()
        print("[INFO] search took {} seconds".format(end - start))

        #so# loop over the results
        if len(results)==0:
            st.write('No hay coincidencias')
        sorted_results = sorted(results)
        for (d, h) in sorted_results:
            # grab all image paths in our dataset with the same hash
            resultPaths = hashes.get(h, [])
            print("[INFO] {} total image(s) with d: {}, h: {}".format(
                len(resultPaths), d, h))

            # loop over the result paths
            for resultPath in resultPaths:
                # load the result image and display it to our screen
                result = cv2.imread(resultPath)
                plt_imshow(f'Distancia: {d} bits',result)
def main():
    #Query
    if(os.path.isfile('vptree.pickle') and os.path.isfile('vptree.pickle')):
        query()
    else:
        train()
if __name__ == '__main__':
    main()