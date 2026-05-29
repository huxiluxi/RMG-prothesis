from matplotlib.legend import Legend
import matplotlib.pyplot as plt

def export_pubfig(fig, filename, width=3.5, aspectRatio=0.75):
    """
    EXPORT_PUBFIG Export a matplotlib figure with publication settings.
    
    Parameters:
        fig      - matplotlib figure object
        filename - output file path (e.g., "figure.pdf")
        width    - width in inches (default 3.5). Height = width * aspectRatio
    """
    # === USER SETTINGS ===
    axisFontSize     = 12
    labelFontSize    = 14
    titleFontSize    = 16
    legendFontSize   = 10
    axisLineWidth    = 1
    plotLineWidth    = 1.5
    tickDirection    = 'out'  # 'in' or 'out'
    boxOn            = False
    gridOn           = False
    legendBoxOn      = True
    maxWidth         = 6.3
    # aspectRatio      = 0.75  # height = width * aspectRatio
    pad_inches       = 0.0  # No padding

    if width > maxWidth:
        print("Warning: Width is larger than 6.3, ensure it fits your page")

    height = width * aspectRatio
    fig.set_size_inches(width, height)

    # Apply styles to all axes in figure
    for ax in fig.get_axes():
        ax.tick_params(labelsize=axisFontSize, direction=tickDirection, width=axisLineWidth)
        for spine in ax.spines.values():
            spine.set_linewidth(axisLineWidth)
        # ax.set_box_aspect(aspectRatio)
        if not boxOn:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        ax.grid(gridOn)

        # Set font sizes for labels and title
        if ax.get_xlabel(): ax.set_xlabel(ax.get_xlabel(), fontsize=labelFontSize)
        if ax.get_ylabel(): ax.set_ylabel(ax.get_ylabel(), fontsize=labelFontSize)
        if ax.get_title():  ax.set_title(ax.get_title(), fontsize=titleFontSize)

    # Style all legends manually
    for ax in fig.get_axes():
        for child in ax.get_children():
            if isinstance(child, Legend):
                for text in child.get_texts():
                    text.set_fontsize(legendFontSize)
                if child.get_title():
                    child.get_title().set_fontsize(legendFontSize)
                child.set_frame_on(legendBoxOn)

    # Style lines
    for line in fig.findobj(match=plt.Line2D):
        line.set_linewidth(plotLineWidth)

    # Save with vector format and tight bounding box
    fig.savefig(filename, bbox_inches='tight', pad_inches=pad_inches, transparent=True)
