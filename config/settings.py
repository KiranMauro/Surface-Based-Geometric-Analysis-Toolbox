from dataclasses import dataclass, field


@dataclass
class Settings:

    # ---------------------------------------------------------
    # Alignment
    # ---------------------------------------------------------

    alignment_plot: bool = False

    # ---------------------------------------------------------
    # Voxel → Vertex
    # ---------------------------------------------------------

    method: str = "B"
    max_dist: float = 2

    # ---------------------------------------------------------
    # Parameter Maps
    # ---------------------------------------------------------

    r2_threshold: float = 0.4

    r2_display_threshold: float = 0.2

    num_range: tuple = (1.1, 7)

    max_eccentricity: float = 8

    # ---------------------------------------------------------
    # Candidate Detection
    # ---------------------------------------------------------

    thmin: float = 0

    min_area: float = 200

    gof_threshold: float = 0.5

    min_fraction: float = 0.5

    # ---------------------------------------------------------
    # Split Suggestion
    # ---------------------------------------------------------
    min_cluster_area: float = 200
    
    min_split_area: float = 50

    split_max_dist: float = 10

    alpha: float = 1.0

    beta: float = 0.3

    min_edge_weight: float = 0.05

    split_similarity_threshold: float = 0.75

    n_clusters: tuple[int, ...] = (2, 3)


    # ---------------------------------------------------------
    # Refinement
    # ---------------------------------------------------------
    default_erode: int = 1

    default_dilate: int = 3

    min_component_area: int = 200
    # ---------------------------------------------------------
    # Ridge_paths
    # ---------------------------------------------------------
    concavity_threshold: float = 0.15

    depth_threshold_sulcus: float = 0.2

    depth_threshold_gyrus: float = 0.6


@dataclass
class Settings:

    processing: Settings = field(
        default_factory=Settings
    )