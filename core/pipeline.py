from processing import (
    alignment,
    voxel2vertex,
    create_parameter_map,
    detect_candidates,
    compute_topographic_directions,
    suggest_splits,
    curate_splits,
    refinement,
    curate_clusters,
    compute_cluster_metadata,
)

from functools import partial

class Pipeline:

    def __init__(self, project):

        self.project = project

        self.steps = [

            ("alignment", alignment.run),

            ("voxel2vertex", voxel2vertex.run),

            ("parameter_maps", create_parameter_map.run),

            ("detect_candidates", detect_candidates.run),

            ("topographic_directions", compute_topographic_directions.run),

            ("suggest_splits", suggest_splits.run),

            ("curate_splits", curate_splits.run),

            ("refinement", refinement.run),

            ("curate_clusters", curate_clusters.run),

            ("compute_metadata", compute_cluster_metadata.run),

        ]

    def run(self):
        for name, step in self.steps:
            self._print_header(name)
            step(self.project)

        print("\nPipeline complete.")

    def run_step(self, step_name):

        for name, step in self.steps:
            if name == step_name:
                self._print_header(name)
                step(self.project)
                return

        raise ValueError( f"Unknown pipeline step '{step_name}'.")
    
    def run_from(self, step_name):
        start = False

        for name, step in self.steps:
            if name == step_name:
                start = True

            if start:
                self._print_header(name)
                step(self.project)

        if not start:
            raise ValueError(f"Unknown pipeline step '{step_name}'.")
        
    def _print_header(self, title):
        print()
        print("=" * 70)
        print(title.replace("_", " ").title())
        print("=" * 70)
        