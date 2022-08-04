from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import colorsys

def add_vertical_lines_annotations(
    ax,
    v: List[Tuple[float, str]],
    colors: Optional[List] = None,
    colormap: str = "Paired",
    text_background_color: str = "w",
):

    if colors is None:
        colors = plt.get_cmap(colormap).colors[-len(v):]
    assert len(v) == len(colors)
    ymin, ymax = ax.set_ylim()
    
    half_height = (ymax-ymin)/2
    lowest_position = ymin + half_height
    for i, ((value, name), color) in enumerate(zip(v, colors)):
        hls = colorsys.rgb_to_hls(*color)
        if hls[1] > 0.5:
            text_background_color='#777777'
        else:
            text_background_color='#FAFAFA'
        
        ax.axvline(x=value, label=f"{name}: {value:.2f}", color=color, linewidth=2.5)
        
        txt = ax.text(x=value, y=lowest_position + (half_height* (i%5)/5), s=f"{name}: {value:.2f}", color=color)
        txt.set_path_effects(
            [PathEffects.withStroke(linewidth=4, foreground=text_background_color)]
        )

    ax.legend()