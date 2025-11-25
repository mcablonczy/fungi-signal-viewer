# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 10:23:26 2025

@author: markablonczy
"""


from typing import Dict, List, Tuple

import matplotlib.cm as cm


Color = Tuple[int, int, int]  # (r, g, b) 0–255


def get_viridis_colors(n: int) -> List[Color]:
    """
    Return n distinct colors from the viridis colormap as (r,g,b) tuples 0–255.
    """
    n = max(0, int(n))
    cmap = cm.get_cmap("viridis", n if n > 0 else 1)

    colors: List[Color] = []
    for i in range(n):
        r, g, b, _ = cmap(i)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def build_channel_color_map(channel_names: List[str]) -> Dict[str, Color]:
    """
    Given a list of channel names, assign each a distinct viridis color.
    """
    viridis_colors = get_viridis_colors(len(channel_names))
    return {
        str(name): viridis_colors[i]
        for i, name in enumerate(channel_names)
    }
