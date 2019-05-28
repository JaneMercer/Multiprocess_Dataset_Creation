import os
import pickle 
import random
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import cv2
import numpy as np


DATADIR = "H:\working_space\python\PetImages"   #path to folders with data
DIRECTORIES = ["Dog", "Cat"]                    #subcategories of data folder
IMG_SIZE = 60
CORES = cpu_count()                             # number of cores you want to be engaged in multiprocessing
                                                # cpu_count() returnes int value of how many cores your computer processor has
METHODS = ["m_simple", "m_dft"]                 # METHODS: 'm_simple' - simply orig.img in grayscale; 'm_dft' - discrete Fourier transform method
I_NDEX = 1                                      # index for array METHODS to choose the method quickly on the start


def m_simple(orig_img):
    gr = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY) #cv2.COLOR_BGR2GRAY - Gray conversion
    # h, s, v = cv2.split(hsv)
    return gr


def m_dft(orig_img):
    g = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(g), flags=cv2.DFT_COMPLEX_OUTPUT) #discrete Fourier transform (DFT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])).astype('uint8')
    return magnitude_spectrum


def process_with(orig_img, method):
    try:
        res = globals()[method](orig_img)
        return res
    except Exception as e:
        print(e)
        pass


#   |prepare_img| takes path to orig image; reads it; chooses method to processes it into desired "form"
#   file_name and path - combine into full path for cv2.imread, label - label for current image, method - method to process with
def prepare_img(file_name, path, label, method):
    try:
        img = cv2.imread(os.path.join(path, file_name), cv2.IMREAD_COLOR)  # gets image with deafault params
        if img is None:
            pass
        else:
            scaled_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # resizes by cropping, by default interpolation=None
            res_image = process_with(scaled_img, method)
            return [res_image, label]
    except Exception as e:
        print(e)
        pass


# """ #-----------------CREATION
if __name__ == '__main__':

    F_features = []
    L_labels = []

    start_t = time.time()  # for time tracking only

    with Pool(CORES) as p:

        training_data = []  # list of tuples: (processed_img,label)

        for directory in DIRECTORIES:  # for each folder under path DATADIR
            path = os.path.join(DATADIR, directory)  # path to images
            class_num = DIRECTORIES.index(directory)  # int - index of current element from DIRECTORIES
            images = os.listdir(path)  # list of filenames

            training_data = p.map(partial(prepare_img, path=path, label=class_num, method=METHODS[I_NDEX]), images) #Pool object parallelizes the execution of a prepare_img() function
                                                                                                                    #collects results into training_data
            training_data_clean = [i for i in training_data if i]  # deletes all "Null Object"s from training_data

            if training_data_clean:
                print(len(training_data_clean))
                training_data = random.shuffle(training_data_clean)
                for features, label in training_data_clean:
                    F_features.append(features)
                    L_labels.append(label)

    end_t = time.time()
    print(f'TIME SPENT: {end_t - start_t:.2f}s \n')

    #F_features = np.array(F_features).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # change 1 for 3  for 3 chanels

    pickle_out = open("F_features.pickle", "wb")  # pickle file for features
    pickle.dump(F_features, pickle_out) #writes a pickled representation of obj to the open file
    pickle_out.close()

    pickle_out2 = open("L_lables.pickle", "wb")  # pickle file for lables
    pickle.dump(L_labels, pickle_out2)
    pickle_out2.close()

"""
# -----------------OPEN CREATED

pickle_in = open("F_features.pickle", "rb")
X = pickle.load(pickle_in)
print(X[0])
Y = pickle.load(open("L_lables.pickle", "rb"))
print(Y[0])
pickle_in.close()
"""
