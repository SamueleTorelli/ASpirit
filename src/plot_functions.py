import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Arc
import plotly.graph_objects as go
import plotly.io as pio

from cycler     import cycler
from contextlib import contextmanager


color_sequence = ("k", "m", "g", "b", "r",
                  "gray", "aqua", "gold", "lime", "purple",
                  "brown", "lawngreen", "tomato", "lightgray", "lightpink")

def auto_plot_style(overrides = dict()):
    plt.rcParams[ "figure.figsize"               ] = 10, 8
    plt.rcParams[   "font.size"                  ] = 20
    plt.rcParams[  "lines.markersize"            ] = 25
    plt.rcParams[  "lines.linewidth"             ] = 3
    plt.rcParams[  "patch.linewidth"             ] = 3
    plt.rcParams[   "axes.linewidth"             ] = 2
    plt.rcParams[   "grid.linewidth"             ] = 3
    plt.rcParams[   "grid.linestyle"             ] = "--"
    plt.rcParams[   "grid.alpha"                 ] = 0.5
    plt.rcParams["savefig.dpi"                   ] = 300
    plt.rcParams["savefig.bbox"                  ] = "tight"
    plt.rcParams[   "axes.formatter.use_mathtext"] = True
    plt.rcParams[   "axes.formatter.limits"      ] = (-3 ,4)
    plt.rcParams[  "xtick.major.size"            ] = 8
    plt.rcParams[  "ytick.major.size"            ] = 8
    plt.rcParams[  "xtick.minor.size"            ] = 5
    plt.rcParams[  "ytick.minor.size"            ] = 5
    plt.rcParams[   "axes.prop_cycle"            ] = cycler(color=color_sequence)
    plt.rcParams[  "image.cmap"                  ] = "gnuplot2"
    plt.rcParams.update(overrides)



@contextmanager
def temporary(name, new_value):
    old_value          = plt.rcParams[name]
    plt.rcParams[name] = new_value
    try    : yield
    finally: plt.rcParams[name] = old_value


def normhist(x, *args, normto=100, normfactor=None, **kwargs):
    if "histtype" not in kwargs:
        kwargs["histtype"] = "step"
    if normfactor is None:
        w = np.full(len(x), normto/len(x))
    else:
        w = np.full(len(x), normfactor)
    return plt.hist(x, *args, weights=w, **kwargs)


def normhist2d(x, y, *args, normto=100, normfactor=None, **kwargs):
    if normfactor is None:
        w = np.full(len(x), normto/len(x))
    else:
        w = np.full(len(x), normfactor)
    if "cmin" not in kwargs:
        kwargs["cmin"] = w[0]
    return plt.hist2d(x, y, *args, weights=w, **kwargs)


def plot_circular_sectors(sector_angle, radial_bins_per_sector, radius=480, center=(0, 0), dpi=180):
    """
    Plot the radial sector division used for the analysis.

    Parameters:
        sector_angle : float
            Angular width of each sector in degrees (e.g., 60 gives 6 sectors).
        radial_bins_per_sector : int
            Number of radial divisions (rings) per sector.
        radius : float
            Maximum radius of the circle.
        center : tuple
            Center of the circle (default is (0, 0)).
        dpi : int
            DPI for the figure.
    """
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    ax.set_aspect('equal')

    # Draw outer circle
    circle = plt.Circle(center, radius, fill=False, color='k', linewidth=2)
    ax.add_patch(circle)

    assert 360 % sector_angle == 0, "sector_angle must divide 360 evenly"
    n_sectors = 360 // sector_angle

    # Draw each sector
    for i in range(n_sectors):
        start_angle = i * sector_angle
        end_angle = (i + 1) * sector_angle
        color = color_sequence[i % len(color_sequence)]

        # Sector wedge
        wedge = Wedge(center, radius, start_angle, end_angle, 
                      facecolor=color, alpha=0.5, edgecolor='k')
        ax.add_patch(wedge)

        # Angle label
        mid_angle = (start_angle + end_angle) / 2
        label_radius = radius * 0.5
        x = label_radius * np.cos(np.radians(mid_angle))
        y = label_radius * np.sin(np.radians(mid_angle))
        ax.text(x, y, f'{start_angle:.0f}°-{end_angle:.0f}°', 
                ha='center', va='center', fontsize=12)

    # Radial lines
    for angle in np.arange(0, 360, sector_angle):
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        ax.plot([0, x], [0, y], 'k-', linewidth=1, alpha=0.5)

    # Radial arcs and labels in first sector only
    radial_bins = np.linspace(0, radius, radial_bins_per_sector + 1)
    for r in radial_bins[1:-1]:
        arc = Arc(center, 2 * r, 2 * r, angle=0,
                  theta1=0, theta2=sector_angle,
                  color='k', linewidth=1, alpha=0.7)
        ax.add_patch(arc)

        # Label inside the first sector
        label_angle = sector_angle / 6  # ~center-ish of first wedge
        x = r * np.cos(np.radians(label_angle))
        y = r * np.sin(np.radians(label_angle))
        ax.text(x, y, f'{r:.0f}', backgroundcolor='white',
                ha='center', va='center', fontsize=10)

    # Final plot formatting
    ax.set_xlim(-radius * 1.1, radius * 1.1)
    ax.set_ylim(-radius * 1.1, radius * 1.1)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    
    plt.tight_layout()
    plt.show()


def plot_3D_points_with_Q(df):
    # Extract arrays
    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()
    z = df["Z"].to_numpy()
    Q = df["Q"].to_numpy()

    # Interactive plot
    pio.renderers.default = "notebook_connected"

    fig = go.Figure()

    # 3D scatter with Q
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=Q,          # color by Q
            colorscale='Viridis',
            colorbar=dict(title="Q"),
        ),
        name='Points (colored by Q)'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=900,
        height=700,
        showlegend=True
    )

    fig.show()