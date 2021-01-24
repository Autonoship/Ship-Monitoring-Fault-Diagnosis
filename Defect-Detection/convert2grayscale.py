import os
import cv2


def grayscale(path):
    img = cv2.imread(path, 1) # read color image
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def get_dir(path):
    files = os.listdir(path)
    dir_list = [f for f in files if os.path.isdir(os.path.join(path, f))]
    return dir_list


def get_file(path):
    files = os.listdir(path)
    file_list = [f for f in files if os.path.isfile(os.path.join(path, f))]
    return file_list


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def bundle_grayscale(path):
    gray_path = 'Gray_' + path
    create_dir(gray_path)       # create a gray image directory

    dir_list = get_dir(path)    # color image directory list
    for d in dir_list:
        out_dir = os.path.join(*[gray_path, d])
        create_dir(out_dir)               # create a gray image directory with the same name for each label

        in_dir = os.path.join(*[path, d]) # color image directory for each label
        file_list = get_file(in_dir)      # color image list
        for f in file_list:
            in_path = os.path.join(*[in_dir, f])   # color image path
            out_path = os.path.join(*[out_dir, f]) # gray  image path
            img = grayscale(in_path)
            cv2.imwrite(out_path, img)


bundle_grayscale('Concrete_Images')
