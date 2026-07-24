from __future__ import annotations
from pathlib import Path
from config.settings import Settings
from core.graph import GraphGeodesics
from data.io import (
    load_clusters,
    load_vertex_data,
    save_clusters,
    save_vertex_data,
)
from core.pipeline import Pipeline


class Project:

    def __init__(self, subject, model):

        self.subject = subject
        self.model = model

        self.settings = Settings()
            
        self._cluster_set = None
        self._vertex_data = None

        self._geo_lh = None
        self._geo_rh = None

    @property
    def project_root(self):
       return Path(__file__).resolve().parents[2]

    @property
    def data_dir(self):
        return self.project_root / "data" / self.subject
    
    @property
    def input_dir(self):
        return self.data_dir / "input"
    
    @property
    def output_dir(self):
        return self.data_dir / "output"
    
    @property
    def parameter_map_dir(self):
        return self.output_dir / "parameter_maps"
    
    @property
    def anatomy_dir(self):
        return self.output_dir / "anatomy"
    
    @property
    def statistics_dir(self):
        return self.output_dir / "statistics"

    @property
    def label_dir(self):
        return self.output_dir / "labels"
    
    @property
    def candidate_dir(self):
        return self.label_dir / "candidates" / self.model
    @property
    def final_dir(self):
        return self.label_dir / "final" / self.model

    @property
    def split_dir(self):
        return self.label_dir / "split" / self.model

    @property
    def deleted_dir(self):
        return self.label_dir / "deleted" / self.model
             
    @property
    def surf_dir(self):
        return self.data_dir / "s1" / "surf"
    
    @property
    def mri_dir(self):
        return self.data_dir / "s1" / "mri"
    
    @property
    def cluster_path(self):
        return self.output_dir / "data_structures" / f"clusters_{self.model}.npy"
        
    @property
    def vertex_data_path(self):
        return self.output_dir / "data_structures" / f"vertex_data_{self.model}.npy"
        
    @property
    def cluster_set(self):
        if self._cluster_set is None:
            self._cluster_set = load_clusters(self.cluster_path)

        return self._cluster_set
        
    @property
    def vertex_data(self):
        if self._vertex_data is None:
            self._vertex_data = load_vertex_data(self.vertex_data_path)

        return self._vertex_data
    
    @property
    def geo_lh(self):
        if self._geo_lh is None:
            self._geo_lh = GraphGeodesics(self.surf_dir / "lh.white")

        return self._geo_lh
    
    @property
    def geo_rh(self):
        if self._geo_rh is None:
            self._geo_rh = GraphGeodesics(self.surf_dir / "rh.white")

        return self._geo_rh
    
    @property
    def geo(self):
        return {"lh": self.geo_lh, "rh": self.geo_rh}
    
    def save_vertex_data(self, vertex_data):
        self._vertex_data = vertex_data
        self.vertex_data_path.parent.mkdir(parents=True, exist_ok=True)
        save_vertex_data(vertex_data, self.vertex_data_path)

    def save_clusters(self, cluster_set):
        self._cluster_set = cluster_set
        self.cluster_path.parent.mkdir(parents=True, exist_ok=True)
        save_clusters(cluster_set, self.cluster_path)

    def run_pipeline(self):
        Pipeline(self).run()

    def run_pipeline_from(self, step):
        Pipeline(self).run_from(step)

    def run_pipeline_step(self, step):
        Pipeline(self).run_step(step)

    
