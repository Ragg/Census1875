# coding=utf-8
import math
import shutil
import os
import errno
import itertools
from collections import defaultdict
import sys

import numpy as np
import cv2
import pypyodbc
import scandir


root_dir = os.getcwd()


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            result = self.default_factory(key)
            if result is None:
                raise KeyError(key)
            else:
                self[key] = result
                return result


def load_template(name):
    return cv2.imread(os.path.join(root_dir, "templates", name),
                      cv2.IMREAD_GRAYSCALE)


templates = keydefaultdict(load_template)


def debug_write_image(name, image):
    if __debug__:
        cv2.imwrite(name, image)


def debug_open_file(name, mode):
    if not __debug__:
        name = os.devnull
    return open(name, mode)


def compute_angle(line):
    angle = math.atan2(line[3] - line[1], line[2] - line[0])
    return angle


def compute_skew(image):
    lines = cv2.HoughLinesP(image, 1, math.pi / 360, 1, None, 30, 1)[0]
    mean_angle = 0
    num_angles = 0
    for line in lines:
        angle = compute_angle(line)
        if abs(abs(angle) - math.pi / 2) > math.degrees(10):
            continue
        if abs(angle) > math.pi / 4:
            if angle > 0:
                angle -= math.pi / 2
            elif angle < 0:
                angle += math.pi / 2
        mean_angle += angle
        num_angles += 1
    mean_angle /= num_angles
    return mean_angle


def rotate(image, angle):
    rows, cols = image.shape
    mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), math.degrees(angle), 1)
    rotated = cv2.warpAffine(image, mat, (cols, rows))
    return rotated


def poly_merge_line(l, image_shape, is_vertical):
    points_x = []
    points_y = []
    for li in l:
        x1 = li[0]
        y1 = li[1]
        x2 = li[2]
        y2 = li[3]
        if is_vertical:
            if y1 > y2:
                start = y2
                stop = y1
            else:
                start = y1
                stop = y2
        else:
            if x1 > x2:
                start = x2
                stop = x1
            else:
                start = x1
                stop = x2
        if is_vertical and x1 == x2:
            for y in xrange(start, stop + 1):
                points_x.append(x1)
                points_y.append(y)
        else:
            a = float((y2 - y1)) / (x2 - x1)
            b = y1 - a * x1
            poly = np.poly1d((a, b))
            if is_vertical:
                for y in xrange(start, stop + 1):
                    x = np.poly1d(poly - y).roots[0]
                    points_x.append(x)
                    points_y.append(y)
            else:
                for x in xrange(start, stop + 1):
                    y = poly(x)
                    points_x.append(x)
                    points_y.append(y)

    p = np.poly1d(np.polyfit(points_x, points_y, 1))
    if is_vertical:
        image_height = image_shape[0]
        point_top = (int(round(p.roots[0])), 0)
        point_bottom = (
            int(round((p - image_height).roots[0])), image_height)
        return point_top + point_bottom
    else:
        image_width = image_shape[1]
        point_left = (0, int(round(p(0))))
        point_right = (image_width, int(round(p(image_width))))
        return point_left + point_right


def simple_merge_line(l, start, end):
    assert start < end
    key = lambda x: x[1]
    s = sorted(l, key=key)
    first = s[0]
    last = s[-1]
    x1 = first[0]
    y1 = first[1]
    x2 = last[2]
    y2 = last[3]
    if x1 == x2:
        point_top = (x1, start)
        point_bottom = (x1, end)
        return point_top + point_bottom
    a = float((y2 - y1)) / (x2 - x1)
    b = y1 - a * x1
    p = np.poly1d((a, b))
    point_top = (int(round((p - start).roots[0])), start)
    point_bottom = (int(round((p - end).roots[0])), end)
    return point_top + point_bottom


def merge_lines(lines, image_shape):
    lines_split = [[], []]
    for l in lines:
        if min(l[1], l[3]) < image_shape[0] / 2:
            lines_split[0].append(l)
        else:
            lines_split[1].append(l)
    key = lambda line: line[0]
    lines_merged = []
    first = True
    for lines_to_merge in lines_split:
        lines_sorted = sorted(lines_to_merge, key=key)
        lines_adjacent_collection = []
        lines_adjacent = [lines_sorted[0]]
        prev = min(lines_sorted[0][0], lines_sorted[0][2])
        merged_lines_text = "mergelinesV.txt"
        min_length = 5
        with debug_open_file(merged_lines_text, "w") as txt:
            l = lines_sorted[0]
            txt.write("{}, {}\n".format((l[0], l[1]), (l[2], l[3])))
            for l in lines_sorted[1:]:
                cur = min(l[0], l[2])
                diff = abs(cur - prev)
                if diff < 9:
                    lines_adjacent.append(l)
                else:
                    if len(lines_adjacent) >= min_length:
                        lines_adjacent_collection.append(lines_adjacent)
                    lines_adjacent = [l]
                prev = cur
                txt.write("{}, {}\n".format((l[0], l[1]), (l[2], l[3])))
            if len(lines_adjacent) >= min_length:
                lines_adjacent_collection.append(lines_adjacent)
            if first:
                start = 0
                end = image_shape[0] / 2
            else:
                start = image_shape[0] / 2
                end = image_shape[0]
            lines_merged.append([simple_merge_line(l, start, end) for l in
                                 lines_adjacent_collection])
            first = False
    final_lines = []
    for x in xrange(0, 3):
        final_lines.append(
            simple_merge_line([lines_merged[0][x], lines_merged[1][x]], 0,
                              image_shape[0]))
    return final_lines


def crop(source, top, bottom, left, right):
    cropped = source[top:bottom, left:right]
    return cropped


def crop_relative(source, top, bottom, left, right):
    w = source.shape[1]
    h = source.shape[0]
    top = int(top * h)
    left = int(left * w)
    return crop(source, top, bottom * h, left, right * w), left, top


def find_template(template, source, top, bottom, left, right):
    h, w = template.shape
    s = source.copy()
    source_cropped, offset_x, offset_y = crop_relative(s, top, bottom, left,
                                                       right)
    debug_write_image("templateregion.png", source_cropped)
    method = cv2.TM_CCOEFF_NORMED
    result = cv2.matchTemplate(source_cropped, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    top_left = (top_left[0] + offset_x, top_left[1] + offset_y)
    bottom_right = (bottom_right[0] + offset_x, bottom_right[1] + offset_y)
    return top_left, bottom_right


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def seg_intersect(la, lb):
    a1 = np.array([float(la[0]), float(la[1])])
    a2 = np.array([float(la[2]), float(la[3])])
    b1 = np.array([float(lb[0]), float(lb[1])])
    b2 = np.array([float(lb[2]), float(lb[3])])
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    result = num / denom * db + b1
    return int(round(result[0])), int(round(result[1]))


def find_lines(image):
    lines_binary = cv2.HoughLinesP(image, 1, math.pi / 360, 20, None, 30, 1)[0]
    lines_vertical = []
    for x in lines_binary:
        if abs(abs(compute_angle(x)) - math.pi / 2) < math.radians(10):
            lines_vertical.append(x)
    lines_horizontal = []
    start = image.shape[0] * 0.077
    step = (image.shape[0] - start) * 0.0526
    for x in xrange(0, 19):
        y = int(start + step * x)
        lines_horizontal.append([0, y, 100, y])
    lines_horizontal.append([0, image.shape[0], 100, image.shape[0]])
    return lines_vertical, lines_horizontal


def find_genders(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    debug_write_image("binary.png", binary)
    lines_vertical, lines_horizontal = find_lines(binary)
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if __debug__:
        lines_img = color.copy()
        for line in itertools.chain(lines_vertical, lines_horizontal):
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            cv2.line(lines_img, pt1, pt2, (0, 0, 255), 2)
        debug_write_image("lines.png", lines_img)
    lines_merged_vertical = merge_lines(lines_vertical, image.shape)
    if len(lines_merged_vertical) != 3:
        print "Vertical lines: {}".format(len(lines_merged_vertical))
    lines_merged_horizontal = [[0, x[1], image.shape[1], x[3]] for x in
                               lines_horizontal]
    if len(lines_merged_horizontal) != 20:
        print "Horizontal lines: {}".format(len(lines_merged_horizontal))
    merged_lines_img = color.copy()
    with debug_open_file("linesV.txt", "w") as line_text:
        for line in lines_merged_vertical:
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            line_text.write("{}, {}\n".format(pt1, pt2))
            cv2.line(merged_lines_img, pt1, pt2, (0, 0, 255), 2)
    with debug_open_file("linesH.txt", "w") as line_text:
        for line in lines_merged_horizontal:
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            line_text.write("{}, {}\n".format(pt1, pt2))
            cv2.line(merged_lines_img, pt1, pt2, (0, 0, 255), 2)
    debug_write_image("linesmerged.png", merged_lines_img)
    intersections = []
    sects = []
    for y in lines_merged_vertical:
        sect = (y[0], y[1])
        sects.append(sect)
        cv2.circle(color, sect, 3, (0, 0, 255), -1)
    intersections.append(sects)
    for x in lines_merged_horizontal:
        sects = []
        for y in lines_merged_vertical:
            sect = seg_intersect(x, y)
            sects.append(sect)
            cv2.circle(color, sect, 3, (0, 0, 255), -1)
        intersections.append(sects)
    debug_write_image("intersections.png", color)
    if len(lines_merged_vertical) != 3:
        return [0]

    genders = []
    for i in xrange(0, len(intersections) - 1):
        cur = intersections[i]
        next = intersections[i + 1]
        male_img = crop(binary, cur[0][1], next[1][1], cur[0][0], next[1][0])
        male_img, _, _ = crop_relative(male_img, 0.05, 0.95, 0.15, 0.85)
        debug_write_image("male.png", male_img)
        num_male = cv2.countNonZero(male_img)
        female_img = crop(binary, cur[1][1], next[2][1], cur[1][0], next[2][0])
        female_img, _, _ = crop_relative(female_img, 0.06, 0.96, 0.15, 0.85)
        debug_write_image("female.png", female_img)
        num_female = cv2.countNonZero(female_img)
        if abs(num_male - num_female) < 50:
            break
        else:
            if num_male > num_female:
                genders.append(1)
            else:
                genders.append(2)
    return genders


def extract_genders(source):
    top_left, bottom_right = find_template(templates["gender_top.png"], source,
                                           0.1, 0.3, 0.3, 0.5)
    crop_top = bottom_right[1]
    margin = (bottom_right[0] - top_left[0]) * 0.3
    crop_left = top_left[0] - margin
    crop_right = bottom_right[0] + margin

    top_left, bottom_right = find_template(templates["gender_bottom.png"],
                                           source, 0.65, 0.85, 0.25, 0.45)
    crop_bottom = top_left[1]
    cropped = crop(source, crop_top, crop_bottom, crop_left, crop_right)
    debug_write_image("cropped.png", cropped)
    return cropped


class ImageCollection(object):
    def __init__(self):
        self.image_dir = r"C:/Users/rhdgjest/Documents/1875/todo"
        self.image_index = {}
        for root, dirs, files in scandir.walk(self.image_dir):
            for f in files:
                self.image_index[f] = os.path.join(root, f)
        self._conn = pypyodbc.connect(
            r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
            r"Dbq=C:\Users\rhdgjest\Documents\censusscan\data\RestVestfold"
            r".accdb;")
        self._query = u"""
        SELECT DISTINCT IMAGE_ID FROM main
        WHERE EVENT_CLERICAL_DISTRICT='{}' AND (MULTI_RECORD_TYPE)='TYPE 1'
        ORDER BY main.IMAGE_ID;"""

    def query(self, district):
        cursor = self._conn.cursor()
        cursor.execute(self._query.format(district))
        for row in cursor:
            image_name = os.path.split(row[0])[1]
            image_path = self.image_index[image_name]
            input_name = image_path
            if __debug__:
                copy_dir = os.path.join(root_dir, "debug", image_name)
                try:
                    os.makedirs(copy_dir)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                os.chdir(copy_dir)
                shutil.copy(image_path, image_name)
                input_name = image_name
            yield image_name, input_name


if __name__ == "__main__":
    def main():
        np.seterr('raise')
        coll = ImageCollection()
        districts = (unicode(x, sys.stdin.encoding) for x in sys.argv[1:])
        districts = [u"Strømm"]
        for district in districts:
            images = coll.query(district)
            gender_collection = []
            for image_name, input_name in images:
                if "00021" not in image_name:continue
                source = cv2.imread(input_name, cv2.IMREAD_GRAYSCALE)
                cropped = extract_genders(source)
                genders = find_genders(cropped)
                gender_string = "{} {}\n".format(image_name,
                    " ".join(str(x) for x in genders))
                print gender_string
                gender_collection.append(gender_string)
            with open(
                    os.path.join(root_dir, u"genders_{}.txt".format(district)),
                    "w") as f:
                f.writelines(gender_collection)

    main()
