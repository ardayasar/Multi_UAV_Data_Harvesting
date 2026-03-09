import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Initialize figure (wider to fit 3 columns cleanly)
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('off')  # Hide axes

# Define legend entries (markers and colors match your image)
legend_elements = [
    mlines.Line2D([], [], color='brown', marker='*', linestyle='None', markersize=10,
                  label='Victim Position'),
    mlines.Line2D([], [], color='orange', marker='^', linestyle='None', markersize=10,
                  label='Anchor Device Position'),
    mpatches.Patch(color='lightgray', label='UAV Start Zone'),
    mpatches.Patch(color='lightblue', label='UAV Terminal Zone'),
    mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8,
                  label="UAV1's Trajectory"),
    mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8,
                  label="UAV2's Trajectory"),
    mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=8,
                  label="UAV3's Trajectory"),
]

# Create legend: 3 columns × 3 rows layout
legend = ax.legend(
    handles=legend_elements,
    loc='center',
    ncol=3,
    frameon=True,
    framealpha=0.9,
    edgecolor='gray',
    fancybox=True,
    handletextpad=1.5,
    columnspacing=1.8
)

plt.tight_layout()
plt.savefig("legend_box_3x3.png", dpi=300, bbox_inches='tight')
plt.show()