from dataclasses import dataclass, field
import numpy as np


@dataclass
class VertexData:
    coords: np.ndarray
    metrics: dict[str, np.ndarray]
    voxel_index: np.ndarray
    metadata: dict = field(default_factory=dict)

    def get_metric(self, name):
        return self.metrics[name]

    def has_metric(self, name):
        return name in self.metrics


@dataclass
class Cluster:
    subject: str
    id: int
    hemi: str
    valid_vertices: np.ndarray
    label_vertices: np.ndarray
    fsaverage_vertices: np.ndarray | None = None
    centroid: int | None = None
    fsaverage_centroid: int | None = None
    parcel: str | None = None
    name: str | None = None
    group_id: int | None = None
    directions: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)

    def __hash__(self):
        return hash((self.subject, self.hemi, self.id))

    def __eq__(self, other):
        return (
            isinstance(other, Cluster)
            and self.subject == other.subject
            and self.hemi == other.hemi
            and self.id == other.id
        )


@dataclass
class ClusterSet:
    candidate_clusters: list[Cluster] = field(default_factory=list)
    final_clusters: list[Cluster] = field(default_factory=list)
    deleted_clusters: list[Cluster] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
@dataclass
class CorrespondenceGroup:
    id: int
    members: dict = field(default_factory=dict)
    statistics: dict = field(default_factory=dict)

    @property
    def clusters(self):
        return list(self.members.values())