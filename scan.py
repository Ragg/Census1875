# coding=utf-8
import math
import shutil
import os
import errno
import itertools
from collections import defaultdict

import numpy as np
import cv2
import pypyodbc


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


def debug_write(name, image):
    if __debug__:
        cv2.imwrite(name, image)


def compute_angle(line):
    angle = math.atan2(line[3] - line[1], line[2] - line[0])
    return angle


def compute_skew(image):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    lines = cv2.HoughLinesP(binary, 1, math.pi / 360, 1, None, 70, 1)[0]
    mean_angle = 0
    with open("angles.txt", "w") as angles:
        for line in lines:
            angle = compute_angle(line)
            if abs(angle) > math.pi / 4:
                if angle > 0:
                    angle -= math.pi / 2
                elif angle < 0:
                    angle += math.pi / 2
            mean_angle += angle
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            angles.write(
                "{}-{}: {}/{}\n".format(pt1, pt2, math.degrees(angle), angle))
    mean_angle /= len(lines)
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


def simple_merge_line(l, image_shape, is_vertical):
    if is_vertical:
        key = lambda x: x[1]
    else:
        key = lambda x: x[0]
    s = sorted(l, key=key)
    first = s[0]
    last = s[-1]
    x1 = first[0]
    y1 = first[1]
    x2 = last[2]
    y2 = last[3]
    if x1 == x2:
        point_top = (x1, 0)
        point_bottom = (x1, image_shape[0])
        return point_top + point_bottom

    a = float((y2 - y1)) / (x2 - x1)
    b = y1 - a * x1
    p = np.poly1d((a, b))
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


def merge_lines(lines_to_merge, image_shape, is_vertical=True):
    if is_vertical:
        key = lambda line: line[0]
    else:
        key = lambda line: line[1]
    lines_sorted = sorted(lines_to_merge, key=key)
    lines_adjacent_collection = []
    lines_adjacent = [lines_sorted[0]]
    if is_vertical:
        prev = min(lines_sorted[0][0], lines_sorted[0][2])
        merged_lines_text = "mergelinesV.txt"
        min_length = 5
    else:
        prev = min(lines_sorted[0][1], lines_sorted[0][3])
        merged_lines_text = "mergelinesH.txt"
        min_length = 1
    with open(merged_lines_text, "w") as txt:
        l = lines_sorted[0]
        txt.write("{}, {}\n".format((l[0], l[1]), (l[2], l[3])))
        for l in lines_sorted[1:]:
            if is_vertical:
                cur = min(l[0], l[2])
            else:
                cur = min(l[1], l[3])
            diff = abs(cur - prev)
            if diff < 9:
                lines_adjacent.append(l)
            else:
                if len(lines_adjacent) >= min_length:
                    lines_adjacent_collection.append(lines_adjacent)
                lines_adjacent = [l]
            prev = cur
            txt.write("{}, {}\n".format((l[0], l[1]), (l[2], l[3])))
        if len(lines_adjacent) > 5:
            lines_adjacent_collection.append(lines_adjacent)
    if is_vertical:
        return [simple_merge_line(l, image_shape, True) for l in
                lines_adjacent_collection]
    else:
        return [poly_merge_line(l, image_shape, False) for l in
                lines_adjacent_collection]


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
    debug_write("templateregion.png", source_cropped)
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


def find_lines(image, lines_img):
    lines = cv2.HoughLinesP(image, 1, math.pi / 360, 1, None, 15, 1)[0]
    print len(lines)
    lines_horizontal = []
    lines_vertical = []
    for x in lines:
        if abs(compute_angle(x)) > math.pi / 4:
            if abs(abs(compute_angle(x)) - math.pi / 2) < math.radians(10):
                lines_vertical.append(x)
        else:
            if abs(compute_angle(x)) < math.radians(10):
                lines_horizontal.append(x)
    for line in itertools.chain(lines_vertical, lines_horizontal):
        pt1 = (line[0], line[1])
        pt2 = (line[2], line[3])
        cv2.line(lines_img, pt1, pt2, (0, 0, 255), 2)
    cv2.imwrite("lines.png", lines_img)
    return lines_vertical, lines_horizontal


def find_genders(image):
    _, binary = cv2.threshold(image, 127, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite("binary.png", binary)
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lines_vertical, lines_horizontal = find_lines(binary, color.copy())
    lines_merged_vertical = merge_lines(lines_vertical, image.shape)
    if len(lines_merged_vertical) != 3:
        print "Vertical lines: {}".format(len(lines_merged_vertical))
        return None
    lines_merged_horizontal = merge_lines(lines_horizontal, image.shape, False)
    merged_lines_img = color.copy()
    with open("linesV.txt", "w") as line_text:
        for line in lines_merged_vertical:
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            line_text.write("{}, {}\n".format(pt1, pt2))
            cv2.line(merged_lines_img, pt1, pt2, (0, 0, 255), 2)
    with open("linesH.txt", "w") as line_text:
        for line in lines_merged_horizontal:
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            line_text.write("{}, {}\n".format(pt1, pt2))
            cv2.line(merged_lines_img, pt1, pt2, (0, 0, 255), 2)
    cv2.imwrite("linesmerged.png", merged_lines_img)
    intersections = []
    sects = []
    for y in lines_merged_vertical:
        sect = (y[0], y[1])
        sects.append(sect)
        cv2.circle(color, sect, 3, (0, 0, 255), -1)
    intersections.append(sects)
    for x in xrange(15):
        sects = []
        for y in lines_merged_vertical:
            sect = seg_intersect(lines_merged_horizontal[x], y)
            sects.append(sect)
            cv2.circle(color, sect, 3, (0, 0, 255), -1)
        intersections.append(sects)
    cv2.imwrite("intersections.png", color)

    genders = []
    for i in xrange(0, len(intersections) - 1):
        cur = intersections[i]
        next = intersections[i + 1]
        male_img = crop(image, cur[0][1], next[1][1], cur[0][0], next[1][0])
        male_img, _, _ = crop_relative(male_img, 0.1, 0.9, 0.1, 0.9)
        _, male_img = cv2.threshold(male_img, 127, 255, cv2.THRESH_BINARY_INV)
        num_male = 0
        for j in male_img.flat:
            if j > 0:
                num_male += 1
        female_img = crop(image, cur[1][1], next[2][1], cur[1][0], next[2][0])
        female_img, _, _ = crop_relative(female_img, 0.1, 0.9, 0.1, 0.9)
        _, female_img = cv2.threshold(female_img, 127, 255,
                                      cv2.THRESH_BINARY_INV)
        num_female = 0
        for j in female_img.flat:
            if j > 0:
                num_female += 1
        if num_male < 80 and num_female < 80:
            genders.append(0)
        else:
            if num_male > num_female:
                genders.append(1)
            else:
                genders.append(2)
    return genders


def extract_genders(source):
    top_left, bottom_right = find_template(template_gender_top, source, 0.1,
                                           0.3, 0.3, 0.5)
    crop_top = bottom_right[1]
    crop_left = top_left[0] - 0.1 * source.shape[0]
    crop_right = top_left[0] + 0.1 * source.shape[1]

    top_left, bottom_right = find_template(template_gender_bottom, source, 0.65,
                                           0.85, 0.25, 0.45)
    crop_bottom = bottom_right[1]
    cropped = crop(source, crop_top, crop_bottom, crop_left, crop_right)
    cv2.imwrite("cropped.png", cropped)

    angle = compute_skew(cropped)
    rotated = rotate(source, angle)

    top_left, bottom_right = find_template(template_gender_top, rotated, 0.1,
                                           0.3, 0.3, 0.5)
    crop_top = bottom_right[1]
    crop_left = top_left[0]
    crop_right = bottom_right[0]
    top_left, bottom_right = find_template(template_gender_bottom, rotated,
                                           0.65, 0.85, 0.25, 0.45)
    crop_bottom = bottom_right[1]
    rotated_cropped = crop(rotated, crop_top, crop_bottom, crop_left,
                           crop_right)
    cv2.imwrite("rotated.png", rotated_cropped)
    return rotated_cropped


def extract_absent(source):
    top_left, bottom_right = find_template(templates["absent_left.png"], source,
                                           0.65, 0.85, 0.05, 0.25)
    crop_left = bottom_right[0]
    crop_bottom = bottom_right[1] - 10
    top_left, bottom_right = find_template(templates["absent_top.png"], source,
                                           0.6, 0.8, 0.15, 0.35)
    crop_top = bottom_right[1]
    top_left, bottom_right = find_template(templates["absent_right.png"],
                                           source, 0.6, 0.8, 0.25, 0.45)
    crop_right = top_left[0] - 5

    cropped = crop(source, crop_top, crop_bottom, crop_left, crop_right)

    debug_write("cropped.png", cropped)
    blurred = cv2.medianBlur(cropped, 5)
    debug_write("blurred.png", blurred)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    debug_write("binary.png", binary)
    return binary


def main():
    np.seterr('raise')
    image_dir = r"C:/Users/rhdgjest/Documents/004706498/"
    working_dir = os.getcwd()

    conn = pypyodbc.connect(
        r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
        r"Dbq=C:\Users\rhdgjest\Documents\censusscan\data\RestVestfold.accdb;")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT IMAGE_ID FROM main "
                   "WHERE FOLDER=4706498 AND MULTI_RECORD_TYPE='TYPE 1' "
                   "ORDER BY IMAGE_ID")
    absent = []
    # for k, v in (i for i in res.iteritems() if '917' in i[0]):
    for row in cursor:
        img = row[0]
        image_split = os.path.split(img)
        image_name = image_split[1]
        image_path = os.path.join(image_dir, image_name)
        input_name = image_path
        if (__debug__):
            copy_dir = os.path.join(working_dir, "results", image_name)
            try:
                os.makedirs(copy_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            os.chdir(copy_dir)
            shutil.copy(image_path, image_name)
            input_name = image_name
        source = cv2.imread(input_name, cv2.IMREAD_GRAYSCALE)
        binary = extract_absent(source)
        if binary is None:
            absent.append((image_name, "?????"))
            print "{}: ?????".format(image_name)
            continue
        pixels = sum(1 for _ in (pix for pix in binary.flat if pix > 0))
        if pixels > 500:
            print "{}:  {}".format(image_name, pixels)
            absent.append((image_name, pixels))

    key = lambda x: x[0]
    absentstrings = ("{}:  {}\n".format(x[0], x[1]) for x in
                     sorted(absent, key=key))
    with open("absent.txt", "w") as out:
        out.writelines(absentstrings)


main()
