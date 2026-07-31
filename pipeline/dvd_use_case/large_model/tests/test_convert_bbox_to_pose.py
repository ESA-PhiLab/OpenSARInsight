from utilities.convert_bbox_to_pose import bbox_to_pose_line

def test_convert_bbox_to_pose():
    bbox = {'left': 10, 'right': 50, 'top': 20, 'bottom': 60}
    class_id = 0
    img_width = 100
    img_height = 100

    result = bbox_to_pose_line(bbox, class_id, img_width, img_height)
    assert isinstance(result, str)
    # Check output format and values
    parts = result.split()
    assert len(parts) == 8
    assert parts[0] == str(class_id)
    # Check that all coordinates are floats and visibility is 2
    for p in parts[1:7]:
        float(p)
    assert parts[7] == '2'