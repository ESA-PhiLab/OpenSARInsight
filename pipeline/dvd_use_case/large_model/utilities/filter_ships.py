"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
filter_ships.py

Tool: Filter ships based on distance and vessel status

Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Creation Date: 2026-02-26

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""
# type: ignore
from typing import Optional, Literal


def filter_ships(
    ships: list,
    distance_threshold_km: Optional[float] = None,
    distance_metric: Literal['xview', 'global', 'both'] = 'xview',
    is_vessel_filter: Optional[bool] = None
) -> list:
    """
    Filter ships based on distance and vessel status.
    
    Args:
        ships: List of ship dictionaries from parse_vessel_xml
        distance_threshold_km: Maximum distance from shore in km (None = no filter)
        distance_metric: Which distance metric to use ('xview', 'global', or 'both')
        is_vessel_filter: Filter by is_vessel status (None = no filter, True/False = filter)
        
    Returns:
        list: Filtered list of ships
    """
    filtered = ships
    
    # Filter by is_vessel if specified
    if is_vessel_filter is not None:
        filtered = [s for s in filtered if s['is_vessel'] == is_vessel_filter]
    
    # Filter by distance if threshold specified (includes ships at threshold distance and above)
    if distance_threshold_km is not None:
        if distance_metric == 'xview':
            filtered = [s for s in filtered if s['xview_distance_km'] >= distance_threshold_km]
        elif distance_metric == 'global':
            filtered = [s for s in filtered if s['global_distance_km'] >= distance_threshold_km]
        elif distance_metric == 'both':
            # Must satisfy both distance thresholds
            filtered = [
                s for s in filtered 
                if s['xview_distance_km'] >= distance_threshold_km 
                and s['global_distance_km'] >= distance_threshold_km
            ]
    
    return filtered