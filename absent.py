"""Attempt to determine which households in a district had persons
temporarily absent by extracting the field from the photograph and analyzing
it"""
import os
import sys

import cv2
import numpy as np

from scan import ImageCollection, root_dir, find_template, templates, crop, \
    debug_write_image


def main():
    np.seterr('raise')
    coll = ImageCollection()
    districts = (unicode(x, sys.stdin.encoding) for x in sys.argv[1:])
    for district in districts:
        images = coll.query(district)
        absent = []
        for image_name, input_name in images:
            source = cv2.imread(input_name, cv2.IMREAD_GRAYSCALE)
            # Do a binarization of the absent field and count the amount of
            # pixels that are not black, and output this score, which will be
            # used as a basis for whether or not there are absent persons
            # listed.
            binary = extract_absent(source)
            if binary is None:
                print "{} error".format(image_name)
                absent.append((image_name, "error"))
                continue
            pixels = (0 < binary).sum()
            absent.append((image_name, pixels))

        key_score = lambda x: x[0]
        absentstrings = ("{} {}\n".format(x[0], x[1]) for x in
                         sorted(absent, key=key_score))
        scorestrings = []
        key_name = lambda x: x[1]
        format_str = '<a href="{}">{}</a><br>'
        for x in sorted(absent, key=key_name):
            scorestrings.append(format_str.format(coll.image_index[x[0]],
                                                  "{} {}\n".format(x[0], x[1])))

        with open(os.path.join(root_dir, u"{}.txt".format(district)), "w") as \
                out:
            out.writelines(absentstrings)
        # Write out the data in HTML form, sorted by the scores, and with
        # links to the images, so that we can manually inspect them later and
        # determine what the cutoff value should be for whether a household
        # had persons temporarily absent or not
        with open(os.path.join(root_dir, u"{}.html".format(district)),
                  "w") as out:
            out.writelines(scorestrings)


def extract_absent(source):
    """Extract the absent field by finding points around it via template
    matching, and using those points to determine the bounds of the absent
    field."""
    top_top_left, top_bottom_right = find_template(templates["absent_top.png"],
                                                   source, 0.65, 0.9, 0.1, 0.4)
    crop_top = top_bottom_right[1]
    left = float(top_bottom_right[0]) / source.shape[1]
    right = left + 0.15
    top = max(0.0, float(top_top_left[1]) / source.shape[0] - 0.04)
    bottom = top + 0.1
    top_left, bottom_right = find_template(templates["absent_right.png"],
                                           source, top, bottom, left, right)
    crop_right = top_left[0] - 5
    right = float(top_top_left[0]) / source.shape[1]
    left = max(0.0, right - 0.1)
    top_left, bottom_right = find_template(templates["absent_left.png"], source,
                                           top, bottom, left, right)
    crop_left = bottom_right[0] + 5
    crop_bottom = crop_top + (bottom_right[1] - crop_top) * 2.5

    cropped = crop(source, crop_top - 50, crop_bottom + 50, crop_left - 50,
                   crop_right + 50)

    debug_write_image("cropped.png", cropped)
    blurred = cv2.GaussianBlur(cropped, (5, 5), 0)
    debug_write_image("blurred.png", blurred)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    debug_write_image("binary.png", binary)
    cropped_binary = crop(binary, 50, binary.shape[0] - 50, 50,
                          binary.shape[1] - 50)
    debug_write_image("cropped_binary.png", cropped_binary)
    return cropped_binary


if __name__ == "__main__":
    main()
