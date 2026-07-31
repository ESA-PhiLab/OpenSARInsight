import tempfile
from pathlib import Path
from utilities.read_xml import parse_vessel_xml

def test_parse_vessel_xml(tmp_path):
		# Create a minimal XML file with 3 ships: valid, missing bbox, NaN bbox
		xml_content = '''
		<Root>
			<Number_of_ships>3</Number_of_ships>
			<Ship>
				<Name>ShipA</Name>
				<BoundingBox>
					<Top>10</Top>
					<Left>20</Left>
					<Bottom>30</Bottom>
					<Right>40</Right>
				</BoundingBox>
				<xView3_shoreline_distance_from_shore_km>5.5</xView3_shoreline_distance_from_shore_km>
				<global_shoreline_vector_distance_from_shore_km>7.7</global_shoreline_vector_distance_from_shore_km>
				<Is_Vessel>true</Is_Vessel>
			</Ship>
			<Ship>
				<Name>ShipB</Name>
				<BoundingBox>
					<Top>nan</Top>
					<Left>20</Left>
					<Bottom>30</Bottom>
					<Right>40</Right>
				</BoundingBox>
				<Is_Vessel>false</Is_Vessel>
			</Ship>
			<Ship>
				<Name>ShipC</Name>
				<!-- Missing BoundingBox -->
			</Ship>
		</Root>
		'''
		xml_file = tmp_path / "test.xml"
		xml_file.write_text(xml_content)

		result = parse_vessel_xml(xml_file)
		assert result['number_of_ships'] == 3
		# Only ShipA should be parsed as valid
		assert len(result['ships']) == 1
		ship = result['ships'][0]
		assert ship['name'] == 'ShipA'
		assert ship['bbox'] == {'top': 10, 'left': 20, 'bottom': 30, 'right': 40}
		assert ship['xview_distance_km'] == 5.5
		assert ship['global_distance_km'] == 7.7
		assert ship['is_vessel'] is True
		# Two ships should be skipped
		assert len(result['skipped_ships']) == 2
