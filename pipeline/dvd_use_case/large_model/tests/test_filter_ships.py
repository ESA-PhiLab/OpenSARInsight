import math

import pytest
from utilities.filter_ships import filter_ships


@pytest.fixture
def ships():
	# Only valid ships for general tests
	return [
		{'is_vessel': True, 'xview_distance_km': 5.0, 'global_distance_km': 10.0},
		{'is_vessel': False, 'xview_distance_km': 15.0, 'global_distance_km': 20.0},
		{'is_vessel': True, 'xview_distance_km': 25.0, 'global_distance_km': 30.0},
		{'is_vessel': False, 'xview_distance_km': 35.0, 'global_distance_km': 40.0},
	]

def test_filter_with_nan():
	ships = [
		{'is_vessel': True, 'xview_distance_km': float('nan'), 'global_distance_km': 50.0},
		{'is_vessel': True, 'xview_distance_km': 60.0, 'global_distance_km': float('nan')},
		{'is_vessel': True, 'xview_distance_km': 5.0, 'global_distance_km': 10.0},
		{'is_vessel': False, 'xview_distance_km': 15.0, 'global_distance_km': 20.0},
		{'is_vessel': True, 'xview_distance_km': 25.0, 'global_distance_km': 30.0},
		{'is_vessel': False, 'xview_distance_km': 35.0, 'global_distance_km': 40.0},
	]
	# Filtering with NaN in xview_distance_km: should exclude NaN when threshold is set
	filtered = filter_ships(ships, distance_threshold_km=10, distance_metric='xview')
	for s in filtered:
		# NaN comparison always False, so only non-NaN and >=10 remain
		assert not math.isnan(s['xview_distance_km'])
		assert s['xview_distance_km'] >= 10

	# Filtering with NaN in global_distance_km: should exclude NaN when threshold is set
	filtered = filter_ships(ships, distance_threshold_km=10, distance_metric='global')
	for s in filtered:
		assert not math.isnan(s['global_distance_km'])
		assert s['global_distance_km'] >= 10

def test_filter_by_is_vessel(ships):
	filtered = filter_ships(ships, is_vessel_filter=True)
	assert all(s['is_vessel'] for s in filtered)
	assert len(filtered) == 2

def test_filter_by_distance_xview(ships):
	filtered = filter_ships(ships, distance_threshold_km=20, distance_metric='xview')
	assert all(s['xview_distance_km'] >= 20 for s in filtered)
	assert len(filtered) == 2

def test_filter_by_distance_global(ships):
    filtered = filter_ships(ships, distance_threshold_km=25, distance_metric='global')
    assert all(s['global_distance_km'] >= 25 for s in filtered)
    assert len(filtered) == 2  # was 3

def test_filter_by_both(ships):
    filtered = filter_ships(ships, distance_threshold_km=30, distance_metric='both')
    assert all(s['xview_distance_km'] >= 30 and s['global_distance_km'] >= 30 for s in filtered)
    assert len(filtered) == 1  # was 2

def test_filter_combined(ships):
	filtered = filter_ships(ships, distance_threshold_km=30, distance_metric='both', is_vessel_filter=False)
	assert all((not s['is_vessel']) and s['xview_distance_km'] >= 30 and s['global_distance_km'] >= 30 for s in filtered)
	assert len(filtered) == 1
