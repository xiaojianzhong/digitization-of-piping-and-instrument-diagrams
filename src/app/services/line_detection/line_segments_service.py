# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from logger_config import get_logger
import time
import cv2
import numpy as np
from app.config import config
from app.models.line_detection.line_segment import LineSegment
from app.models.bounding_box import BoundingBox
from typing import Optional
from app.utils.image_utils import is_data_element_within_bounding_box

logger = get_logger(__name__)


def detect_line_segments(pid_id: str,
                         preprocessed_image: np.ndarray,
                         image_height: int,
                         image_width: int,
                         max_line_gap: int,
                         threshold: int,
                         min_line_length: int,
                         rho: float,
                         theta_param: float,
                         bounding_box_inclusive:
                             Optional[BoundingBox],
                         ) -> list[LineSegment]:

    """
    Detects the line segments in the image using the text detection
    results and request params
    pid_id: The pid id
    preprocessed_image: The preprocessed image
    image_height: The image height
    image_width: The image width
    bounding_box_inclusive: The bounding box to reduce noise in this detection phase
    within which the lines will be returned
    max_line_gap: The maximum allowed gap between line segments
    to treat them as a single line
    threshold: The accumulator threshold parameter
    min_line_length: The min line length
    rho: The distance resolution of the accumulator in pixels
    theta_param: The angle resolution of the accumulator in radians
    :return: A list of line segments
    """
    logger.info('Starting to detect line segments using Hough transform')

    start = time.perf_counter()

    # Apply the Hough line transform to detect lines
    # rho: Distance resolution of the accumulator in pixels.
    # theta: Angle resolution of the accumulator in radians.
    # threshold: Accumulator threshold parameter.
    # Only those lines are returned that get enough votes ( >threshold ).
    # minLineLength: Minimum line length.
    # Line segments shorter than this are rejected.
    # maxLineGap: Maximum allowed gap between line segments
    # to treat them as a single line.
    hough_results_line_segments = cv2.HoughLinesP(
        preprocessed_image, rho=rho,
        theta=np.pi/theta_param,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    output_line_segments = []

    for line in hough_results_line_segments:
        x1, y1, x2, y2 = line[0]

        '''sorting start and end points of the line segment such that
        left most point is start and right most is end for horizontal lines
        and top most point is start and bottom most is end for vertical lines
        this will help with line flow'''
        # Check if the line segment is horizontal
        if y1 == y2:
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
        # Check if the line segment is vertical
        elif x1 == x2:
            if y1 > y2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
        # The line segment is angled
        else:
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            elif x1 == x2 and y1 > y2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1

        '''include lines that are within defined
        bounding box's coordinates (topX, topY, bottomX and bottomY)
        to avoid noise for line detection'''
        if is_data_element_within_bounding_box(bounding_box_inclusive, x1, y1, x2, y2):
            # Add the detected line to the list of line segments
            # and also normalise the coordinates
            output_line_segments.append(LineSegment(
                startX=x1/image_width,
                startY=y1/image_height,
                endX=x2/image_width,
                endY=y2/image_height
            ))

    end = time.perf_counter()

    logger.info('Completed detecting line segments after {:.4f} seconds'
                .format(end - start))

    return output_line_segments


if __name__ == "__main__":
    import argparse
    import os
    import json
    import xml.etree.ElementTree as ET
    from app.services.line_detection.utils.line_detection_image_preprocessor import LineDetectionImagePreprocessor

    parser = argparse.ArgumentParser(
        description='Run line segment detection on the given image.')
    parser.add_argument(
        "--pid-id",
        type=str,
        dest="pid_id",
        help="The pid id",
        required=True
    )
    parser.add_argument(
        "--image-path",
        type=str,
        dest="image_path",
        help="The path to the image",
        required=True
    )
    parser.add_argument(
        "--symbol-detection-results-path",
        type=str,
        dest="symbol_detection_results_path",
        help="The path to the symbol detection results",
        required=True
    )
    parser.add_argument(
        "--text-detection-results-path",
        type=str,
        dest="text_detection_results_path",
        help="The path to the text detection results",
        required=True
    )
    parser.add_argument(
        "--relevant-bounding-box-for-detection",
        dest="bounding_box_inclusive",
        type=json.loads,
        help="coordinates to exclude legend and outerbox border lines"
    )

    args = parser.parse_args()

    pid_id = args.pid_id
    image_path = args.image_path
    symbol_detection_results_path = args.symbol_detection_results_path
    text_detection_results_path = args.text_detection_results_path
    bounding_box_inclusive = \
        args.bounding_box_inclusive

    if bounding_box_inclusive is not None:
        bounding_box_inclusive = BoundingBox(
            topX=bounding_box_inclusive["topX"],
            topY=bounding_box_inclusive["topY"],
            bottomX=bounding_box_inclusive["bottomX"],
            bottomY=bounding_box_inclusive["bottomY"]
        )
    else:
        bounding_box_inclusive = None

    if not os.path.exists(image_path):
        raise ValueError(f"Image path {image_path} does not exist")

    if not os.path.exists(symbol_detection_results_path):
        raise ValueError(f"Symbol detection results path {symbol_detection_results_path} does not exist")

    if not os.path.exists(text_detection_results_path):
        raise ValueError(f"Text detection results path {text_detection_results_path} does not exist")

    # get bytes from image_path file
    with open(image_path, "rb") as file:
        image_bytes = file.read()
    image_height, image_width = cv2.imread(image_path, cv2.IMREAD_COLOR).shape[:2]

    symbol_bboxes: list[BoundingBox] = []
    tree = ET.parse(symbol_detection_results_path)
    for obj in tree.getroot().findall('object'):
        symbol_bboxes.append(BoundingBox(
            topX=int(obj.find('bndbox').find('xmin').text),
            topY=int(obj.find('bndbox').find('ymin').text),
            bottomX=int(obj.find('bndbox').find('xmax').text),
            bottomY=int(obj.find('bndbox').find('ymax').text),
        ))
    symbol_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    for bbox in symbol_bboxes:
        xmin = int(bbox.topX)
        ymin = int(bbox.topY)
        xmax = int(bbox.bottomX)
        ymax = int(bbox.bottomY)
        cv2.rectangle(symbol_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), 'input', 'symbol_detection_results.jpg'), symbol_image)

    text_bboxes: list[BoundingBox] = []
    tree = ET.parse(text_detection_results_path)
    for obj in tree.getroot().findall('object'):
        text_bboxes.append(BoundingBox(
            topX=int(obj.find('bndbox').find('xmin').text),
            topY=int(obj.find('bndbox').find('ymin').text),
            bottomX=int(obj.find('bndbox').find('xmax').text),
            bottomY=int(obj.find('bndbox').find('ymax').text),
        ))
    text_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    for bbox in text_bboxes:
        xmin = int(bbox.topX)
        ymin = int(bbox.topY)
        xmax = int(bbox.bottomX)
        ymax = int(bbox.bottomY)
        cv2.rectangle(text_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), 'input', 'text_detection_results.jpg'), text_image)

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    image = LineDetectionImagePreprocessor.clear_bounding_boxes(image, symbol_bboxes)

    image = LineDetectionImagePreprocessor.clear_bounding_boxes(image, text_bboxes)

    image = cv2.cvtColor(image,
                         cv2.COLOR_BGR2GRAY)

    image = cv2.threshold(image, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    image = LineDetectionImagePreprocessor.apply_thinning(image)

    lines_list = detect_line_segments(
        pid_id, image,
        image_height=image_height,
        image_width=image_width,
        max_line_gap=config.line_detection_hough_max_line_gap,
        threshold=config.line_detection_hough_threshold,
        min_line_length=config.line_detection_hough_min_line_length,
        rho=config.line_detection_hough_rho,
        theta_param=config.line_detection_hough_theta,
        bounding_box_inclusive=bounding_box_inclusive,
    )

    line_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    for line in lines_list:
        xmin = int(line.startX * image_width)
        ymin = int(line.startY * image_height)
        xmax = int(line.endX * image_width)
        ymax = int(line.endY * image_height)
        cv2.line(line_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), 'output', 'line_detection_results.jpg'), line_image)

    results_output_path = os.path.join(os.path.dirname(__file__), 'output',
                                       'line_detection_results.json')
    # write lines_list to json file
    with open(results_output_path, 'w') as f:
        json.dump([line.__dict__ for line in lines_list], f, indent=4)
