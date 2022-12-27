import os
import sys
import random
import cv2
from pathlib import Path
from collections import defaultdict
import shutil

IMAGES_PATH = "./BelgiumTSD_images"
ANNOTATIONS_PATH = "./BelgiumTSD_annotations"

train_gt = open(os.path.join(ANNOTATIONS_PATH, "BTSD_training_GT.txt"), "r")
test_gt = open(os.path.join(ANNOTATIONS_PATH, "BTSD_testing_GT.txt"), "r")

train_annotation_dict = defaultdict(list)
test_annotation_dict = defaultdict(list)
valid_annotation_dict = defaultdict(list)

random.seed('ilovepwr')

def generate_labels(gt_file, annotation_dict):
    for line in gt_file.readlines():
        line = line.split(";")
        file_path = line[0]
        x, y, x2, y2 = float(line[1]), float(line[2]), float(line[3]), float(line[4])
        # TODO class
        class_label = int(line[6])+1

        height, width = 1236, 1628

        # calculate relative center width and height of BB
        xr = x / width
        yr = y / height
        wr = (x2 - x) / width
        hr = (y2 - y) / height

        x_center = xr + wr / 2.0
        y_center = yr + hr / 2.0

        annotation_dict[os.path.join(IMAGES_PATH, file_path)].append(
            (class_label, x_center, y_center, wr, hr)
        )


print("Generating labels for training set...")
generate_labels(train_gt, train_annotation_dict)
print("Generating labels for testing set...")
generate_labels(test_gt, test_annotation_dict)

print("Splitting testing set to validation and testing set...")
# randomly select half of test set for validation set
valid_keys = random.sample(
    list(test_annotation_dict.keys()), int(0.5 * len(test_annotation_dict))
)

# Remove those labels from test set and add them to validation set
for key in valid_keys:
    value = test_annotation_dict.pop(key)
    valid_annotation_dict[key] = value

TRAINBG_PATH = "./NonTSImages/TrainingBG"
TESTBG_PATH = "./NonTSImages/TestingBG"

print("Adding BG images to sets...")
# to keep ratio of background/object images 1:1 we limit the amount of background images loaded
backgrounds_train = [os.path.join(TRAINBG_PATH, f) for f in os.listdir(TRAINBG_PATH)][
    : len(train_annotation_dict)
]
backgrounds_test = [os.path.join(TESTBG_PATH, f) for f in os.listdir(TESTBG_PATH)][
    : len(test_annotation_dict) + len(valid_annotation_dict)
]
random.shuffle(backgrounds_test)
mi = int(len(backgrounds_test) / 2)
backgrounds_valid = backgrounds_test[:mi]
backgrounds_test = backgrounds_test[mi:]

for bg in backgrounds_train:
    train_annotation_dict[bg] = []

for bg in backgrounds_valid:
    valid_annotation_dict[bg] = []

for bg in backgrounds_test:
    test_annotation_dict[bg] = []

print("Make directories for images and labels...")
# Path("./annotations/train").mkdir(parents=True, exist_ok=True)
# Path("./annotations/test").mkdir(parents=True, exist_ok=True)
# Path("./annotations/valid").mkdir(parents=True, exist_ok=True)
Path("./images/train").mkdir(parents=True, exist_ok=True)
Path("./images/test").mkdir(parents=True, exist_ok=True)
Path("./images/valid").mkdir(parents=True, exist_ok=True)


def save_set(dictionary, collection_file, images_path, annotations_path):
    with open(collection_file, "w+") as collection_f:
        for key in dictionary:
            values = dictionary[key]
            file_name = os.path.basename(key)
            new_file_path = os.path.join(images_path, file_name).replace('.jp2', '.jpg')

            # shutil.copyfile(key, new_file_path)
            image = cv2.imread(key)
            cv2.imwrite(new_file_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

            collection_f.write("{}\n".format(new_file_path))

            label_name = os.path.splitext(file_name)[0] + ".txt"
            label_path = os.path.join(annotations_path, label_name)

            with open(label_path, "w+") as label_file:
                for element in values:
                    # (class_label, x_center, y_center, wr, hr)
                    label_file.write(
                        "{} {} {} {} {}\n".format(
                            element[0], element[1], element[2], element[3], element[4]
                        )
                    )


print("Saving training set...")
save_set(train_annotation_dict, "train.txt", "./images/train", "./images/train")
print("Saving validation set...")
save_set(valid_annotation_dict, "valid.txt", "./images/valid", "./images/valid")
print("Saving testing set...")
save_set(test_annotation_dict, "test.txt", "./images/test", "./images/test")

print("Done!")
