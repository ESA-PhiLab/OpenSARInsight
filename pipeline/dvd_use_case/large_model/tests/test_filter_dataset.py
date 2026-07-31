
import pytest
import re
import sys
import types

# Import functions directly from the script
from scripts.filter_dataset import build_allowed_prefixes, matches_allowed

def test_build_allowed_prefixes_basic():
	labels = ['VD_9_VV', 'VD_8', 'VD_10_VH', 'VD_11_V']
	prefixes = build_allowed_prefixes(labels)
	# VD_9_VV → DB_OPENSAR_VV_VD_9
	# VD_8 → DB_OPENSAR_VV_VD_8 and DB_OPENSAR_VH_VD_8
	# VD_10_VH → DB_OPENSAR_VH_VD_10
	# VD_11_V → DB_OPENSAR_VV_VD_11 and DB_OPENSAR_VH_VD_11
	expected = {
		'DB_OPENSAR_VV_VD_9',
		'DB_OPENSAR_VV_VD_8', 'DB_OPENSAR_VH_VD_8',
		'DB_OPENSAR_VH_VD_10',
		'DB_OPENSAR_VV_VD_11', 'DB_OPENSAR_VH_VD_11',
	}
	assert prefixes == expected

def test_build_allowed_prefixes_invalid_label():
	labels = ['INVALID_LABEL', 'VD_12']
	prefixes = build_allowed_prefixes(labels)
	# Should skip invalid and still add both VV/VH for VD_12
	assert 'DB_OPENSAR_VV_VD_12' in prefixes
	assert 'DB_OPENSAR_VH_VD_12' in prefixes
	assert all(p.startswith('DB_OPENSAR_') for p in prefixes)

def test_matches_allowed_exact_and_aug():
	allowed = {'DB_OPENSAR_VV_VD_9'}
	assert matches_allowed('DB_OPENSAR_VV_VD_9.png', allowed)
	assert matches_allowed('DB_OPENSAR_VV_VD_9_aug1.png', allowed)
	assert not matches_allowed('DB_OPENSAR_VV_VD_10.png', allowed)

def test_matches_allowed_txt_and_other_ext():
	allowed = {'DB_OPENSAR_VH_VD_8'}
	assert matches_allowed('DB_OPENSAR_VH_VD_8.txt', allowed)
	assert matches_allowed('DB_OPENSAR_VH_VD_8_aug2.txt', allowed)
	assert not matches_allowed('DB_OPENSAR_VH_VD_9.txt', allowed)
