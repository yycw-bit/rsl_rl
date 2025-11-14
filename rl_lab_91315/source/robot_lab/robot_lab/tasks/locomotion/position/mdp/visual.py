
import isaaclab.sim as sim_utils
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

GOAL_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cuboid": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
)

POS_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cuboid": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)