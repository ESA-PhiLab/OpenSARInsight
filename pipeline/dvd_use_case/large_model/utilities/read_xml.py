"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
read_xml.py

Tool: Parse vessel XML annotation files

Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Creation Date: 2026-02-26

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""
# type: ignore
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_vessel_xml(xml_path: Path, bbox_tag: str = 'BoundingBox'):
    """
    Parse XML file and extract vessel information.
    
    Args:
        xml_path: Path to XML file
        bbox_tag: XML tag name for the bounding box element
                  (e.g. 'BoundingBox', 'BoundingBox_RC_Local')
        
    Returns:
        dict: Dictionary containing number of ships and ship details
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get number of ships
    num_ships = int(root.find('.//Number_of_ships').text)
    
    # Extract ship details
    ships = []
    skipped_ships = []  # Track ships skipped due to NaN values
    for ship in root.findall('.//Ship'):
        ship_name = ship.find('Name').text

        # Check that the bbox tag exists for this ship
        top_el = ship.find(f'.//{bbox_tag}/Top')
        left_el = ship.find(f'.//{bbox_tag}/Left')
        bottom_el = ship.find(f'.//{bbox_tag}/Bottom')
        right_el = ship.find(f'.//{bbox_tag}/Right')

        if any(el is None for el in [top_el, left_el, bottom_el, right_el]):
            skipped_ships.append({
                'name': ship_name,
                'reason': f'Missing {bbox_tag} element'
            })
            continue

        top_text = top_el.text
        left_text = left_el.text
        bottom_text = bottom_el.text
        right_text = right_el.text

        # Skip ships with NaN bounding box values
        if any(v.lower() == 'nan' for v in [top_text, left_text, bottom_text, right_text]):
            skipped_ships.append({
                'name': ship_name,
                'reason': 'NaN bounding box values'
            })
            continue
        
        xview_el = ship.find('.//xView3_shoreline_distance_from_shore_km')
        global_el = ship.find('.//global_shoreline_vector_distance_from_shore_km')
        is_vessel_el = ship.find('.//Is_Vessel')

        ship_data = {
            'name': ship_name,
            'bbox': {
                'top': int(float(top_text)),
                'left': int(float(left_text)),
                'bottom': int(float(bottom_text)),
                'right': int(float(right_text))
            },
            'xview_distance_km': float(xview_el.text) if xview_el is not None and xview_el.text else None,
            'global_distance_km': float(global_el.text) if global_el is not None and global_el.text else None,
            'is_vessel': is_vessel_el.text.lower() == 'true' if is_vessel_el is not None and is_vessel_el.text else None
        }
        ships.append(ship_data)
    
    return {
        'number_of_ships': num_ships,
        'ships': ships,
        'skipped_ships': skipped_ships
    }


if __name__ == "__main__":
    # Example usage
    xml_file = "path/to/your/file.xml"
    result = parse_vessel_xml(xml_file)
    
    print(f"Number of ships: {result['number_of_ships']}")
    print("\nShip Details:")
    for ship in result['ships']:
        print(f"\n{ship['name']}:")
        print(f"  BBox: {ship['bbox']}")
        print(f"  xView Distance: {ship['xview_distance_km']} km")
        print(f"  Global Distance: {ship['global_distance_km']} km")
        print(f"  Is Vessel: {ship['is_vessel']}")
