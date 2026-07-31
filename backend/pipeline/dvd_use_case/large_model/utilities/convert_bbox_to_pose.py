"""
YOLO-pose label line generator

Author: Abdulhameed Yunusa (ABHY)

This utility provides a single function to convert a bounding box dict to a YOLO-pose label line.
"""

def bbox_to_pose_line(bbox, class_id, img_width, img_height):
    """
    Convert a bounding box dict to a YOLO-pose label line.
    Args:
        bbox: dict with 'left', 'right', 'top', 'bottom'
        class_id: int
        img_width: int
        img_height: int
    Returns:
        str: YOLO-pose label line
    """
    x_center = (bbox['left'] + bbox['right']) / 2.0 / img_width
    y_center = (bbox['top'] + bbox['bottom']) / 2.0 / img_height
    width = (bbox['right'] - bbox['left']) / img_width
    height = (bbox['bottom'] - bbox['top']) / img_height
    # Keypoint = bbox center, visibility = 2
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {x_center:.6f} {y_center:.6f} 2"
