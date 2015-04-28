import math
import numpy as np
import scan
import sys
import cv2


def main():
    np.seterr("raise")
    collection = scan.ImageCollection()
    districts = (unicode(x, sys.stdin.encoding) for x in sys.argv[1:])
    for district in districts:
        images = collection.query(district)
        for image_name, image_path in images:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            top_left, bottom_right = scan.find_template(scan.templates[
                "topleft.png"], image, 0.1, 0.3, 0.1, 0.2)
            left = top_left[0]
            top = bottom_right[1]
            top_left, _ = scan.find_template(scan.templates[
                "absent_left.png"], image, 0.6, 0.8, 0.1, 0.2)
            right = top_left[0]
            bottom = top_left[1]
            cropped = scan.crop(image, top, bottom, left, right)
            scan.debug_write_image("cropped.png", cropped)
            blur = cv2.medianBlur(cropped, 5)
            edges = cv2.Canny(blur, 80, 120)
            scan.debug_write_image("canny.png", edges)
            lines = cv2.HoughLinesP(edges, 1, math.pi/360, 20, None, 20, 1)
            lines_image = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
            for line in lines[0]:
                pt1 = (line[0], line[1])
                pt2 = (line[2], line[3])
                cv2.line(lines_image, pt1, pt2, (0, 0, 255), 1)
            scan.debug_write_image("lines.png", lines_image)
            pass
if __name__ == "__main__":
    main()
