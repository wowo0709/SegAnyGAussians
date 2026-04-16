import os

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
from hdbscan import HDBSCAN

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    from plyfile import PlyData
except ImportError:
    PlyData = None


HDBSCAN_REFINED_CONF_THRESHOLD = 0.60
HDBSCAN_REFINED_MIN_MERGE_SCORE = 0.15
HDBSCAN_REFINED_MIN_MERGE_MARGIN = 0.05


def _normalize_rows(array, eps=1e-6):
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def _compute_local_scale(neighbor_distances, local_scale_neighbors=4, min_scale=1e-3):
    if neighbor_distances.size == 0:
        return np.full((neighbor_distances.shape[0],), min_scale, dtype=np.float32)
    local_k = min(local_scale_neighbors, neighbor_distances.shape[1])
    local = np.median(neighbor_distances[:, :local_k], axis=1)
    valid = np.isfinite(local) & (local > 0)
    base = max(float(np.median(local[valid])), min_scale) if np.any(valid) else min_scale
    local = np.where(valid, np.maximum(local, base), base)
    return local.astype(np.float32, copy=False)


def _compute_sh0_color_weight(anchor_rgb, neighbor_rgb, sh0_color_weight, sh0_color_sigma):
    if anchor_rgb is None or neighbor_rgb is None or float(sh0_color_weight) <= 0.0:
        return None
    denom = max(2.0 * float(sh0_color_sigma) * float(sh0_color_sigma), 1e-6)
    diff = anchor_rgb[:, None, :] - neighbor_rgb
    dist2 = np.sum(diff * diff, axis=-1)
    return np.exp(-float(sh0_color_weight) * dist2 / denom).astype(np.float32, copy=False)


def resolve_optional_mesh_path(model_path=None, mesh_path=None):
    candidates = []
    if mesh_path:
        mesh_path = os.path.expanduser(str(mesh_path))
        if os.path.exists(mesh_path):
            return mesh_path
        return None

    if model_path:
        model_path = os.path.expanduser(str(model_path))
        candidates.extend([
            os.path.join(model_path, 'mesh_learnable_sdf.ply'),
            os.path.join(model_path, 'mesh_integration_sdf.ply'),
            os.path.join(model_path, 'mesh_depth_fusion_sdf.ply'),
        ])
        for iteration_dir in sorted(
            [entry for entry in os.listdir(os.path.join(model_path, 'point_cloud'))] if os.path.isdir(os.path.join(model_path, 'point_cloud')) else [],
            reverse=True,
        ):
            iteration_root = os.path.join(model_path, 'point_cloud', iteration_dir)
            candidates.extend([
                os.path.join(iteration_root, 'mesh_learnable_sdf.ply'),
                os.path.join(iteration_root, 'mesh_integration_sdf.ply'),
                os.path.join(iteration_root, 'mesh_depth_fusion_sdf.ply'),
            ])

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _build_mesh_vertex_adjacency(num_vertices, faces):
    if num_vertices <= 0 or faces.size == 0:
        return csr_matrix((num_vertices, num_vertices), dtype=np.uint8)

    rows = []
    cols = []
    for face in faces.astype(np.int64, copy=False):
        a, b, c = [int(v) for v in face]
        rows.extend([a, b, b, c, c, a])
        cols.extend([b, a, c, b, a, c])
    values = np.ones(len(rows), dtype=np.uint8)
    adjacency = csr_matrix((values, (rows, cols)), shape=(num_vertices, num_vertices), dtype=np.uint8)
    adjacency = adjacency.maximum(adjacency.transpose())
    if num_vertices > 0:
        diag = csr_matrix((np.ones(num_vertices, dtype=np.uint8), (np.arange(num_vertices), np.arange(num_vertices))), shape=(num_vertices, num_vertices))
        adjacency = adjacency.maximum(diag)
    adjacency.eliminate_zeros()
    return adjacency


def _load_mesh_vertices_faces(mesh_path):
    if trimesh is not None:
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            raise ValueError(f'Loaded mesh at {mesh_path} does not expose vertices/faces.')
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        return vertices, faces

    if PlyData is None:
        raise ImportError('Mesh-aware clustering requires either trimesh or plyfile to read mesh PLY files.')

    ply = PlyData.read(mesh_path)
    if 'vertex' not in ply or 'face' not in ply:
        raise ValueError(f'Mesh at {mesh_path} must contain vertex and face elements.')

    vertex_data = ply['vertex'].data
    vertices = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=1).astype(np.float32, copy=False)

    face_element = ply['face']
    if len(face_element.properties) == 0:
        raise ValueError(f'Mesh at {mesh_path} has no face index property.')
    face_prop_name = face_element.properties[0].name
    raw_faces = face_element.data[face_prop_name]
    faces = np.asarray([np.asarray(face, dtype=np.int64) for face in raw_faces if len(face) == 3], dtype=np.int64)
    return vertices, faces


def load_mesh_prior_for_points(xyz, model_path=None, mesh_path=None):
    resolved_mesh_path = resolve_optional_mesh_path(model_path=model_path, mesh_path=mesh_path)
    if resolved_mesh_path is None:
        return None

    vertices, faces = _load_mesh_vertices_faces(resolved_mesh_path)
    if vertices.ndim != 2 or vertices.shape[0] == 0:
        raise ValueError(f'Mesh at {resolved_mesh_path} has no vertices.')
    if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] != 3:
        raise ValueError(f'Mesh at {resolved_mesh_path} does not contain triangular faces.')

    tree = cKDTree(vertices)
    _, point_mesh_vertex_idx = tree.query(np.asarray(xyz, dtype=np.float32), k=1, workers=-1)
    point_mesh_vertex_idx = np.asarray(point_mesh_vertex_idx, dtype=np.int32)
    mesh_vertex_adjacency = _build_mesh_vertex_adjacency(vertices.shape[0], faces)
    return {
        'mesh_path': resolved_mesh_path,
        'point_mesh_vertex_idx': point_mesh_vertex_idx,
        'mesh_vertex_adjacency': mesh_vertex_adjacency,
    }


def _compute_mesh_adjacency_multiplier(anchor_vertex_idx, neighbor_vertex_idx, mesh_vertex_adjacency, mesh_weight):
    if mesh_vertex_adjacency is None or anchor_vertex_idx is None or neighbor_vertex_idx is None or float(mesh_weight) <= 0.0:
        return None

    anchor_vertex_idx = np.asarray(anchor_vertex_idx, dtype=np.int64)
    neighbor_vertex_idx = np.asarray(neighbor_vertex_idx, dtype=np.int64)
    multiplier = np.ones(neighbor_vertex_idx.shape, dtype=np.float32)
    boost = 1.0 + float(mesh_weight)

    for row_idx, anchor_vertex in enumerate(anchor_vertex_idx.tolist()):
        if anchor_vertex < 0 or anchor_vertex >= mesh_vertex_adjacency.shape[0]:
            continue
        start = mesh_vertex_adjacency.indptr[anchor_vertex]
        end = mesh_vertex_adjacency.indptr[anchor_vertex + 1]
        adjacent_vertices = mesh_vertex_adjacency.indices[start:end]
        if adjacent_vertices.size == 0:
            linked = neighbor_vertex_idx[row_idx] == anchor_vertex
        else:
            linked = np.isin(neighbor_vertex_idx[row_idx], adjacent_vertices, assume_unique=False)
        multiplier[row_idx, linked] = boost
    return multiplier


def build_affinity_graph(
    xyz,
    features,
    k=16,
    feature_temperature=0.2,
    spatial_scale=2.5,
    spatial_weight=1.0,
    local_scale_neighbors=4,
    sh0_rgb=None,
    sh0_color_weight=0.0,
    sh0_color_sigma=0.25,
    point_mesh_vertex_idx=None,
    mesh_vertex_adjacency=None,
    mesh_weight=0.0,
):
    xyz = np.asarray(xyz, dtype=np.float32)
    features = _normalize_rows(np.asarray(features, dtype=np.float32))
    if sh0_rgb is not None:
        sh0_rgb = np.asarray(sh0_rgb, dtype=np.float32)
    if point_mesh_vertex_idx is not None:
        point_mesh_vertex_idx = np.asarray(point_mesh_vertex_idx, dtype=np.int32)
    num_points = xyz.shape[0]
    if num_points == 0:
        return csr_matrix((0, 0), dtype=np.float32)
    if num_points == 1:
        return csr_matrix((1, 1), dtype=np.float32)

    tree = cKDTree(xyz)
    query_k = min(k + 1, num_points)
    distances, neighbors = tree.query(xyz, k=query_k, workers=-1)
    if distances.ndim == 1:
        distances = distances[:, None]
        neighbors = neighbors[:, None]

    neighbor_distances = distances[:, 1:].astype(np.float32, copy=False)
    neighbor_indices = neighbors[:, 1:].astype(np.int32, copy=False)
    local_scale = _compute_local_scale(neighbor_distances, local_scale_neighbors=local_scale_neighbors)
    if neighbor_indices.size == 0:
        return csr_matrix((num_points, num_points), dtype=np.float32)

    neighbor_scale = local_scale[np.clip(neighbor_indices, 0, num_points - 1)]
    sigma = np.maximum(np.maximum(local_scale[:, None], neighbor_scale), 1e-3) * float(spatial_scale)
    if float(spatial_weight) > 0.0:
        spatial_term = np.exp(-0.5 * float(spatial_weight) * (neighbor_distances / sigma) ** 2)
    else:
        spatial_term = np.ones_like(neighbor_distances, dtype=np.float32)

    feature_similarity = np.clip(np.sum(features[:, None, :] * features[neighbor_indices], axis=-1), -1.0, 1.0)
    feature_weight = np.exp((feature_similarity - 1.0) / max(float(feature_temperature), 1e-6))

    weights = spatial_term * feature_weight
    if sh0_rgb is not None and float(sh0_color_weight) > 0.0:
        if sh0_rgb.shape[0] != num_points:
            raise ValueError(f'SH0 RGB count ({sh0_rgb.shape[0]}) does not match point count ({num_points}).')
        color_weight = _compute_sh0_color_weight(sh0_rgb, sh0_rgb[neighbor_indices], sh0_color_weight, sh0_color_sigma)
        weights *= color_weight

    if point_mesh_vertex_idx is not None and mesh_vertex_adjacency is not None and float(mesh_weight) > 0.0:
        if point_mesh_vertex_idx.shape[0] != num_points:
            raise ValueError(
                f'Mesh vertex assignment count ({point_mesh_vertex_idx.shape[0]}) does not match point count ({num_points}).'
            )
        mesh_multiplier = _compute_mesh_adjacency_multiplier(
            point_mesh_vertex_idx,
            point_mesh_vertex_idx[neighbor_indices],
            mesh_vertex_adjacency,
            mesh_weight,
        )
        weights *= mesh_multiplier

    valid = np.isfinite(neighbor_distances) & (neighbor_indices >= 0) & np.isfinite(weights) & (weights > 1e-6)
    if not np.any(valid):
        return csr_matrix((num_points, num_points), dtype=np.float32)

    rows = np.repeat(np.arange(num_points, dtype=np.int32), neighbor_indices.shape[1])[valid.reshape(-1)]
    cols = neighbor_indices.reshape(-1)[valid.reshape(-1)]
    vals = weights.reshape(-1)[valid.reshape(-1)].astype(np.float32, copy=False)
    graph = csr_matrix((vals, (rows, cols)), shape=(num_points, num_points), dtype=np.float32)
    graph = graph.maximum(graph.transpose())
    graph.eliminate_zeros()
    return graph


def _normalized_cut_score(graph, left, right):
    cut = float(graph[left][:, right].sum())
    assoc_left = float(graph[left].sum())
    assoc_right = float(graph[right].sum())
    if assoc_left <= 1e-8 or assoc_right <= 1e-8:
        return np.inf
    return cut / assoc_left + cut / assoc_right


def _propose_ncut_split(subgraph, min_cluster_size, cut_threshold):
    num_nodes = subgraph.shape[0]
    if num_nodes < max(2 * int(min_cluster_size), 4):
        return None

    component_count, component_labels = connected_components(subgraph, directed=False, return_labels=True)
    if component_count > 1:
        component_sizes = np.bincount(component_labels)
        primary_component = int(component_sizes.argmax())
        left = np.nonzero(component_labels == primary_component)[0]
        right = np.nonzero(component_labels != primary_component)[0]
        if left.size >= min_cluster_size and right.size >= min_cluster_size:
            return {'score': 0.0, 'left': left, 'right': right}
        return None

    degrees = np.asarray(subgraph.sum(axis=1)).reshape(-1)
    if np.count_nonzero(degrees > 1e-8) < 2:
        return None

    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-8))
    laplacian = diags(np.ones(num_nodes, dtype=np.float32)) - diags(d_inv_sqrt) @ subgraph @ diags(d_inv_sqrt)
    try:
        _, eigenvectors = eigsh(laplacian, k=2, which='SM', tol=1e-3)
    except Exception:
        return None

    fiedler = np.asarray(eigenvectors[:, 1]).reshape(-1)
    candidate_thresholds = np.unique(np.quantile(fiedler, [0.25, 0.5, 0.75]))
    candidate_thresholds = np.concatenate([candidate_thresholds, np.array([0.0], dtype=np.float32)])

    best = None
    for threshold in np.unique(candidate_thresholds):
        left = np.nonzero(fiedler <= threshold)[0]
        right = np.nonzero(fiedler > threshold)[0]
        if left.size < min_cluster_size or right.size < min_cluster_size:
            continue
        score = _normalized_cut_score(subgraph, left, right)
        if not np.isfinite(score):
            continue
        if best is None or score < best['score']:
            best = {'score': float(score), 'left': left, 'right': right}

    if best is None or best['score'] > float(cut_threshold):
        return None
    return best


def recursive_normalized_cut(graph, max_clusters=24, min_cluster_size=128, cut_threshold=0.12):
    num_nodes = graph.shape[0]
    if num_nodes == 0:
        return np.zeros((0,), dtype=np.int64)

    clusters = [np.arange(num_nodes, dtype=np.int64)]
    while len(clusters) < int(max_clusters):
        best_cluster_idx = None
        best_proposal = None
        for cluster_idx, node_indices in enumerate(clusters):
            proposal = _propose_ncut_split(graph[node_indices][:, node_indices], min_cluster_size, cut_threshold)
            if proposal is None:
                continue
            if best_proposal is None or proposal['score'] < best_proposal['score']:
                best_cluster_idx = cluster_idx
                best_proposal = proposal
        if best_cluster_idx is None or best_proposal is None:
            break

        node_indices = clusters.pop(best_cluster_idx)
        clusters.append(node_indices[best_proposal['left']])
        clusters.append(node_indices[best_proposal['right']])

    clusters.sort(key=lambda indices: (-indices.shape[0], int(indices[0]) if indices.size > 0 else -1))
    labels = np.full((num_nodes,), -1, dtype=np.int64)
    for label, node_indices in enumerate(clusters):
        labels[node_indices] = label
    return labels


def _compute_centers(features, labels, num_clusters):
    centers = np.zeros((num_clusters, features.shape[1]), dtype=np.float32)
    for cluster_id in range(num_clusters):
        mask = labels == cluster_id
        if np.any(mask):
            centers[cluster_id] = features[mask].mean(axis=0)
    return _normalize_rows(centers)


def _relabel_contiguous(labels):
    unique_labels = np.unique(labels)
    mapping = {int(label): idx for idx, label in enumerate(unique_labels.tolist())}
    relabeled = np.vectorize(lambda label: mapping[int(label)], otypes=[np.int64])(labels)
    return relabeled.astype(np.int64, copy=False)


def _relabel_nonnegative_contiguous(labels):
    labels = np.asarray(labels, dtype=np.int64)
    valid = labels >= 0
    if not np.any(valid):
        return np.full(labels.shape, -1, dtype=np.int64)
    unique_labels = np.unique(labels[valid])
    relabeled = np.full(labels.shape, -1, dtype=np.int64)
    for new_label, old_label in enumerate(unique_labels.tolist()):
        relabeled[labels == int(old_label)] = int(new_label)
    return relabeled.astype(np.int64, copy=False)


def _counts_from_valid_labels(labels):
    labels = np.asarray(labels, dtype=np.int64)
    valid = labels >= 0
    if not np.any(valid):
        return np.zeros((0,), dtype=np.int64)
    return np.bincount(labels[valid]).astype(np.int64, copy=False)


def _build_knn_connectivity(neighbor_indices, num_points):
    if num_points <= 0 or neighbor_indices.size == 0:
        return csr_matrix((num_points, num_points), dtype=np.uint8)
    rows = np.repeat(np.arange(num_points, dtype=np.int32), neighbor_indices.shape[1])
    cols = neighbor_indices.reshape(-1).astype(np.int32, copy=False)
    valid = cols >= 0
    if not np.any(valid):
        return csr_matrix((num_points, num_points), dtype=np.uint8)
    values = np.ones(int(np.count_nonzero(valid)), dtype=np.uint8)
    graph = csr_matrix((values, (rows[valid], cols[valid])), shape=(num_points, num_points), dtype=np.uint8)
    graph = graph.maximum(graph.transpose())
    graph.eliminate_zeros()
    return graph


def _compute_knn_neighbor_terms(
    xyz,
    k,
    spatial_scale=2.5,
    spatial_weight=1.0,
    local_scale_neighbors=4,
):
    xyz = np.asarray(xyz, dtype=np.float32)
    num_points = xyz.shape[0]
    if num_points == 0:
        empty = np.zeros((0, 0), dtype=np.float32)
        return (
            np.zeros((0, 0), dtype=np.int32),
            empty,
            empty,
            csr_matrix((0, 0), dtype=np.uint8),
        )

    tree = cKDTree(xyz)
    query_k = min(max(int(k), 1) + 1, num_points)
    distances, neighbors = tree.query(xyz, k=query_k, workers=-1)
    if distances.ndim == 1:
        distances = distances[:, None]
        neighbors = neighbors[:, None]

    neighbor_distances = distances[:, 1:].astype(np.float32, copy=False)
    neighbor_indices = neighbors[:, 1:].astype(np.int32, copy=False)
    if neighbor_indices.size == 0:
        return (
            neighbor_indices,
            neighbor_distances,
            np.zeros_like(neighbor_distances, dtype=np.float32),
            csr_matrix((num_points, num_points), dtype=np.uint8),
        )

    local_scale = _compute_local_scale(neighbor_distances, local_scale_neighbors=local_scale_neighbors)
    neighbor_scale = local_scale[np.clip(neighbor_indices, 0, num_points - 1)]
    sigma = np.maximum(np.maximum(local_scale[:, None], neighbor_scale), 1e-3) * float(spatial_scale)
    if float(spatial_weight) > 0.0:
        spatial_term = np.exp(-0.5 * float(spatial_weight) * (neighbor_distances / sigma) ** 2).astype(np.float32, copy=False)
    else:
        spatial_term = np.ones_like(neighbor_distances, dtype=np.float32)

    connectivity_graph = _build_knn_connectivity(neighbor_indices, num_points)
    return neighbor_indices, neighbor_distances, spatial_term, connectivity_graph


def _sample_point_indices(num_points, sample_size, min_points=1024, random_state=0):
    if num_points <= 0:
        return np.zeros((0,), dtype=np.int64)
    sample_count = min(num_points, max(int(min_points), int(sample_size)))
    sample_indices = np.arange(num_points, dtype=np.int64)
    if sample_count < num_points:
        rng = np.random.default_rng(int(random_state))
        sample_indices = np.sort(rng.choice(num_points, size=sample_count, replace=False).astype(np.int64, copy=False))
    return sample_indices.astype(np.int64, copy=False)


def cluster_gaussians_hdbscan(
    features,
    sample_size=20000,
    hdbscan_min_cluster_size=10,
    hdbscan_epsilon=0.01,
    random_state=0,
):
    features = _normalize_rows(np.asarray(features, dtype=np.float32))
    num_points = features.shape[0]
    if num_points == 0:
        return {
            'labels': np.zeros((0,), dtype=np.int64),
            'confidence': np.zeros((0,), dtype=np.float32),
            'counts': np.zeros((0,), dtype=np.int64),
            'num_clusters': 0,
            'sample_indices': np.zeros((0,), dtype=np.int64),
            'method': 'HDBSCAN',
        }

    sample_indices = _sample_point_indices(num_points, sample_size=sample_size, random_state=random_state)
    sampled_features = features[sample_indices]

    clusterer = HDBSCAN(
        min_cluster_size=int(hdbscan_min_cluster_size),
        cluster_selection_epsilon=float(hdbscan_epsilon),
        allow_single_cluster=False,
    )
    sampled_cluster_labels = clusterer.fit_predict(sampled_features)
    unique_labels = [int(label) for label in np.unique(sampled_cluster_labels) if label >= 0]

    if len(unique_labels) == 0:
        return {
            'labels': np.full(num_points, -1, dtype=np.int64),
            'confidence': np.zeros(num_points, dtype=np.float32),
            'counts': np.zeros((0,), dtype=np.int64),
            'num_clusters': 0,
            'sample_indices': sample_indices.astype(np.int64, copy=False),
            'sampled_cluster_labels': sampled_cluster_labels.astype(np.int64, copy=False),
            'method': 'HDBSCAN',
        }

    centers = []
    for label in unique_labels:
        center = sampled_features[sampled_cluster_labels == label].mean(axis=0)
        centers.append(center)
    centers = _normalize_rows(np.stack(centers, axis=0).astype(np.float32, copy=False))

    score = features @ centers.T
    labels = score.argmax(axis=1).astype(np.int64, copy=False)
    confidence = score.max(axis=1).astype(np.float32, copy=False)
    counts = np.bincount(labels, minlength=centers.shape[0]).astype(np.int64, copy=False)
    return {
        'labels': labels,
        'confidence': confidence,
        'counts': counts,
        'num_clusters': int(centers.shape[0]),
        'sample_indices': sample_indices.astype(np.int64, copy=False),
        'sampled_cluster_labels': sampled_cluster_labels.astype(np.int64, copy=False),
        'centers': centers.astype(np.float32, copy=False),
        'method': 'HDBSCAN',
    }


def _compute_core_centers(features, labels, core_mask):
    valid_labels = np.unique(labels[labels >= 0])
    if valid_labels.size == 0:
        return np.zeros((0, features.shape[1]), dtype=np.float32), np.zeros((0,), dtype=bool)
    num_clusters = int(valid_labels.max()) + 1
    centers = np.zeros((num_clusters, features.shape[1]), dtype=np.float32)
    available = np.zeros((num_clusters,), dtype=bool)
    for label in valid_labels.tolist():
        mask = (labels == int(label)) & core_mask
        if not np.any(mask):
            continue
        centers[int(label)] = features[mask].mean(axis=0)
        available[int(label)] = True
    if np.any(available):
        centers[available] = _normalize_rows(centers[available])
    return centers.astype(np.float32, copy=False), available


def _resolve_residue_merge_targets(
    component_indices,
    labels,
    confidence,
    features,
    neighbor_indices,
    spatial_term,
    core_mask,
    core_centers,
    core_available,
    sh0_rgb=None,
    sh0_color_weight=0.0,
    sh0_color_sigma=0.25,
):
    component_indices = np.asarray(component_indices, dtype=np.int64)
    if component_indices.size == 0:
        return None, 0.0, 0.0

    comp_neighbors = neighbor_indices[component_indices]
    if comp_neighbors.size == 0:
        return None, 0.0, 0.0

    comp_neighbor_labels = labels[np.clip(comp_neighbors, 0, labels.shape[0] - 1)]
    edge_mask = (comp_neighbors >= 0) & (comp_neighbor_labels >= 0) & core_mask[np.clip(comp_neighbors, 0, labels.shape[0] - 1)]
    if not np.any(edge_mask):
        return None, 0.0, 0.0

    comp_center = _normalize_rows(features[component_indices].mean(axis=0, keepdims=True))[0]
    total_core_edges = float(np.count_nonzero(edge_mask))

    color_weight = None
    if sh0_rgb is not None and float(sh0_color_weight) > 0.0:
        color_weight = _compute_sh0_color_weight(
            sh0_rgb[component_indices],
            sh0_rgb[np.clip(comp_neighbors, 0, labels.shape[0] - 1)],
            sh0_color_weight,
            sh0_color_sigma,
        )

    candidate_scores = []
    for target_label in np.unique(comp_neighbor_labels[edge_mask]).tolist():
        target_label = int(target_label)
        if target_label < 0 or target_label >= core_available.shape[0] or not core_available[target_label]:
            continue
        target_edges = edge_mask & (comp_neighbor_labels == target_label)
        if not np.any(target_edges):
            continue

        contact_score = float(np.count_nonzero(target_edges) / max(total_core_edges, 1.0))
        feature_score = float(np.clip((float(np.dot(comp_center, core_centers[target_label])) + 1.0) * 0.5, 0.0, 1.0))
        spatial_score = float(np.mean(spatial_term[component_indices][target_edges])) if spatial_term.size > 0 else 1.0

        if color_weight is not None:
            color_score = float(np.mean(color_weight[target_edges]))
            merge_score = 0.35 * contact_score + 0.30 * feature_score + 0.15 * spatial_score + 0.20 * color_score
        else:
            merge_score = 0.45 * contact_score + 0.35 * feature_score + 0.20 * spatial_score
        candidate_scores.append((target_label, float(merge_score)))

    if not candidate_scores:
        return None, 0.0, 0.0

    candidate_scores.sort(key=lambda item: item[1], reverse=True)
    best_label, best_score = candidate_scores[0]
    second_score = candidate_scores[1][1] if len(candidate_scores) > 1 else 0.0
    return int(best_label), float(best_score), float(second_score)


def cluster_gaussians_hdbscan_refined(
    xyz,
    features,
    sample_size=20000,
    hdbscan_min_cluster_size=10,
    hdbscan_epsilon=0.01,
    graph_k=16,
    min_cluster_size=128,
    spatial_scale=2.5,
    spatial_weight=1.0,
    local_scale_neighbors=4,
    sh0_rgb=None,
    sh0_color_weight=0.0,
    sh0_color_sigma=0.25,
    random_state=0,
):
    xyz = np.asarray(xyz, dtype=np.float32)
    features = _normalize_rows(np.asarray(features, dtype=np.float32))
    num_points = features.shape[0]
    if sh0_rgb is not None:
        sh0_rgb = np.asarray(sh0_rgb, dtype=np.float32)
        if sh0_rgb.shape[0] != num_points:
            raise ValueError(f'SH0 RGB count ({sh0_rgb.shape[0]}) does not match point count ({num_points}).')
    if float(sh0_color_weight) > 0.0 and sh0_rgb is None:
        raise ValueError('SH0 color weighting was requested for HDBSCANRefined, but no SH0 RGB array was provided.')

    base = cluster_gaussians_hdbscan(
        features,
        sample_size=sample_size,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_epsilon=hdbscan_epsilon,
        random_state=random_state,
    )
    labels = base['labels'].astype(np.int64, copy=True)
    confidence = base['confidence'].astype(np.float32, copy=True)
    if num_points <= 1 or base['num_clusters'] <= 0:
        base['method'] = 'HDBSCANRefined'
        return base

    neighbor_indices, _, spatial_term, connectivity_graph = _compute_knn_neighbor_terms(
        xyz,
        k=graph_k,
        spatial_scale=spatial_scale,
        spatial_weight=spatial_weight,
        local_scale_neighbors=local_scale_neighbors,
    )
    if connectivity_graph.shape[0] == 0:
        base['method'] = 'HDBSCANRefined'
        return base

    core_mask = np.zeros((num_points,), dtype=bool)
    residue_components = []
    for label in np.unique(labels[labels >= 0]).tolist():
        label_idx = np.flatnonzero(labels == int(label))
        if label_idx.size == 0:
            continue
        subgraph = connectivity_graph[label_idx][:, label_idx]
        component_count, component_labels = connected_components(subgraph, directed=False, return_labels=True)
        component_sizes = np.bincount(component_labels)
        core_component = int(component_sizes.argmax())
        core_indices = label_idx[component_labels == core_component]
        core_mask[core_indices] = True

        for component_id in range(component_count):
            if component_id == core_component:
                continue
            component_indices = label_idx[component_labels == component_id]
            mean_confidence = float(confidence[component_indices].mean()) if component_indices.size > 0 else 0.0
            if component_indices.size < int(min_cluster_size) or mean_confidence < HDBSCAN_REFINED_CONF_THRESHOLD:
                residue_components.append(component_indices.astype(np.int64, copy=False))

    core_centers, core_available = _compute_core_centers(features, labels, core_mask)
    for component_indices in residue_components:
        target_label, best_score, second_score = _resolve_residue_merge_targets(
            component_indices,
            labels,
            confidence,
            features,
            neighbor_indices,
            spatial_term,
            core_mask,
            core_centers,
            core_available,
            sh0_rgb=sh0_rgb,
            sh0_color_weight=sh0_color_weight,
            sh0_color_sigma=sh0_color_sigma,
        )
        if (
            target_label is not None
            and best_score >= HDBSCAN_REFINED_MIN_MERGE_SCORE
            and (best_score - second_score) >= HDBSCAN_REFINED_MIN_MERGE_MARGIN
        ):
            labels[component_indices] = int(target_label)
            confidence[component_indices] = float(best_score)
        else:
            labels[component_indices] = -1
            confidence[component_indices] = 0.0

    labels = _relabel_nonnegative_contiguous(labels)
    counts = _counts_from_valid_labels(labels)
    return {
        'labels': labels.astype(np.int64, copy=False),
        'confidence': confidence.astype(np.float32, copy=False),
        'counts': counts.astype(np.int64, copy=False),
        'num_clusters': int(counts.shape[0]),
        'sample_indices': base['sample_indices'],
        'sampled_cluster_labels': base.get('sampled_cluster_labels'),
        'method': 'HDBSCANRefined',
    }


def _merge_small_clusters(features, labels, min_cluster_size):
    labels = labels.astype(np.int64, copy=False)
    counts = np.bincount(labels)
    small_clusters = np.nonzero((counts > 0) & (counts < int(min_cluster_size)))[0]
    if small_clusters.size == 0:
        return labels, counts.astype(np.int64, copy=False)

    large_clusters = np.nonzero(counts >= int(min_cluster_size))[0]
    if large_clusters.size == 0:
        merged = np.zeros_like(labels, dtype=np.int64)
        return merged, np.array([labels.shape[0]], dtype=np.int64)

    large_centers = _compute_centers(features, labels, int(counts.shape[0]))[large_clusters]
    for cluster_id in small_clusters.tolist():
        mask = labels == int(cluster_id)
        if not np.any(mask):
            continue
        similarity = features[mask] @ large_centers.T
        labels[mask] = large_clusters[np.argmax(similarity, axis=1)]

    labels = _relabel_contiguous(labels)
    counts = np.bincount(labels).astype(np.int64, copy=False)
    return labels, counts


def _propagate_labels(
    full_xyz,
    full_features,
    sampled_xyz,
    sampled_features,
    sampled_labels,
    num_clusters,
    propagation_k=16,
    feature_temperature=0.2,
    spatial_scale=2.5,
    spatial_weight=1.0,
    full_sh0_rgb=None,
    sampled_sh0_rgb=None,
    sh0_color_weight=0.0,
    sh0_color_sigma=0.25,
    full_point_mesh_vertex_idx=None,
    sampled_point_mesh_vertex_idx=None,
    mesh_vertex_adjacency=None,
    mesh_weight=0.0,
):
    tree = cKDTree(sampled_xyz)
    chunk_size = 65536
    labels = np.zeros((full_xyz.shape[0],), dtype=np.int64)
    confidence = np.zeros((full_xyz.shape[0],), dtype=np.float32)

    for start in range(0, full_xyz.shape[0], chunk_size):
        end = min(start + chunk_size, full_xyz.shape[0])
        xyz_chunk = full_xyz[start:end]
        feature_chunk = full_features[start:end]
        sh0_chunk = None if full_sh0_rgb is None else full_sh0_rgb[start:end]
        mesh_chunk = None if full_point_mesh_vertex_idx is None else full_point_mesh_vertex_idx[start:end]

        distances, neighbors = tree.query(xyz_chunk, k=min(propagation_k, sampled_xyz.shape[0]), workers=-1)
        if distances.ndim == 1:
            distances = distances[:, None]
            neighbors = neighbors[:, None]

        neighbor_features = sampled_features[neighbors]
        neighbor_labels = sampled_labels[neighbors]
        feature_similarity = np.clip(np.sum(feature_chunk[:, None, :] * neighbor_features, axis=-1), -1.0, 1.0)
        feature_weight = np.exp((feature_similarity - 1.0) / max(float(feature_temperature), 1e-6))

        local_scale = _compute_local_scale(distances.astype(np.float32, copy=False), local_scale_neighbors=min(4, distances.shape[1]))
        sigma = np.maximum(local_scale[:, None] * float(spatial_scale), 1e-3)
        if float(spatial_weight) > 0.0:
            spatial_term = np.exp(-0.5 * float(spatial_weight) * (distances / sigma) ** 2)
        else:
            spatial_term = np.ones_like(distances, dtype=np.float32)
        vote = feature_weight * spatial_term
        if sh0_chunk is not None and sampled_sh0_rgb is not None and float(sh0_color_weight) > 0.0:
            color_weight = _compute_sh0_color_weight(sh0_chunk, sampled_sh0_rgb[neighbors], sh0_color_weight, sh0_color_sigma)
            vote = vote * color_weight
        if mesh_chunk is not None and sampled_point_mesh_vertex_idx is not None and mesh_vertex_adjacency is not None and float(mesh_weight) > 0.0:
            mesh_multiplier = _compute_mesh_adjacency_multiplier(
                mesh_chunk,
                sampled_point_mesh_vertex_idx[neighbors],
                mesh_vertex_adjacency,
                mesh_weight,
            )
            vote = vote * mesh_multiplier
        vote = vote.astype(np.float32, copy=False)

        score = np.zeros((end - start, num_clusters), dtype=np.float32)
        row_index = np.arange(end - start)
        for neighbor_slot in range(neighbors.shape[1]):
            np.add.at(score, (row_index, neighbor_labels[:, neighbor_slot]), vote[:, neighbor_slot])

        labels[start:end] = score.argmax(axis=1).astype(np.int64, copy=False)
        confidence[start:end] = (score.max(axis=1) / np.maximum(score.sum(axis=1), 1e-6)).astype(np.float32, copy=False)

    return labels, confidence


def cluster_gaussians_normalized_cut(
    xyz,
    features,
    sample_size=20000,
    graph_k=16,
    propagation_k=16,
    max_clusters=24,
    min_cluster_size=128,
    cut_threshold=0.12,
    feature_temperature=0.2,
    spatial_scale=2.5,
    spatial_weight=1.0,
    local_scale_neighbors=4,
    sh0_rgb=None,
    sh0_color_weight=0.0,
    sh0_color_sigma=0.25,
    point_mesh_vertex_idx=None,
    mesh_vertex_adjacency=None,
    mesh_weight=0.0,
    random_state=0,
):
    xyz = np.asarray(xyz, dtype=np.float32)
    features = _normalize_rows(np.asarray(features, dtype=np.float32))
    num_points = xyz.shape[0]

    if num_points == 0:
        return {
            'labels': np.zeros((0,), dtype=np.int64),
            'confidence': np.zeros((0,), dtype=np.float32),
            'counts': np.zeros((0,), dtype=np.int64),
            'num_clusters': 0,
        }

    if num_points == 1:
        return {
            'labels': np.zeros((1,), dtype=np.int64),
            'confidence': np.ones((1,), dtype=np.float32),
            'counts': np.array([1], dtype=np.int64),
            'num_clusters': 1,
        }

    if sh0_rgb is not None and sh0_rgb.shape[0] != num_points:
        raise ValueError(f'SH0 RGB count ({sh0_rgb.shape[0]}) does not match point count ({num_points}).')
    if float(sh0_color_weight) > 0.0 and sh0_rgb is None:
        raise ValueError('SH0 color weighting was requested for NormalizedCut, but no SH0 RGB array was provided.')
    if point_mesh_vertex_idx is not None and point_mesh_vertex_idx.shape[0] != num_points:
        raise ValueError(
            f'Mesh vertex assignment count ({point_mesh_vertex_idx.shape[0]}) does not match point count ({num_points}).'
        )

    rng = np.random.default_rng(int(random_state))
    sample_count = min(int(sample_size), num_points)
    sample_indices = np.arange(num_points, dtype=np.int64)
    if sample_count < num_points:
        sample_indices = np.sort(rng.choice(num_points, size=sample_count, replace=False).astype(np.int64, copy=False))

    sampled_xyz = xyz[sample_indices]
    sampled_features = features[sample_indices]
    sampled_sh0_rgb = None if sh0_rgb is None else sh0_rgb[sample_indices]
    sampled_point_mesh_vertex_idx = None if point_mesh_vertex_idx is None else point_mesh_vertex_idx[sample_indices]

    graph = build_affinity_graph(
        sampled_xyz,
        sampled_features,
        k=graph_k,
        feature_temperature=feature_temperature,
        spatial_scale=spatial_scale,
        spatial_weight=spatial_weight,
        local_scale_neighbors=local_scale_neighbors,
        sh0_rgb=sampled_sh0_rgb,
        sh0_color_weight=sh0_color_weight,
        sh0_color_sigma=sh0_color_sigma,
        point_mesh_vertex_idx=sampled_point_mesh_vertex_idx,
        mesh_vertex_adjacency=mesh_vertex_adjacency,
        mesh_weight=mesh_weight,
    )
    sampled_labels = recursive_normalized_cut(
        graph,
        max_clusters=max_clusters,
        min_cluster_size=min_cluster_size,
        cut_threshold=cut_threshold,
    )
    sampled_labels, _ = _merge_small_clusters(sampled_features, sampled_labels, min_cluster_size)
    num_clusters = int(np.max(sampled_labels)) + 1 if sampled_labels.size > 0 else 0
    if num_clusters <= 0:
        return {
            'labels': np.full((num_points,), -1, dtype=np.int64),
            'confidence': np.zeros((num_points,), dtype=np.float32),
            'counts': np.zeros((0,), dtype=np.int64),
            'num_clusters': 0,
        }

    full_labels, confidence = _propagate_labels(
        xyz,
        features,
        sampled_xyz,
        sampled_features,
        sampled_labels,
        num_clusters=num_clusters,
        propagation_k=propagation_k,
        feature_temperature=feature_temperature,
        spatial_scale=spatial_scale,
        spatial_weight=spatial_weight,
        full_sh0_rgb=sh0_rgb,
        sampled_sh0_rgb=sampled_sh0_rgb,
        sh0_color_weight=sh0_color_weight,
        sh0_color_sigma=sh0_color_sigma,
        full_point_mesh_vertex_idx=point_mesh_vertex_idx,
        sampled_point_mesh_vertex_idx=sampled_point_mesh_vertex_idx,
        mesh_vertex_adjacency=mesh_vertex_adjacency,
        mesh_weight=mesh_weight,
    )
    full_labels, counts = _merge_small_clusters(features, full_labels, min_cluster_size)
    return {
        'labels': full_labels.astype(np.int64, copy=False),
        'confidence': confidence.astype(np.float32, copy=False),
        'counts': counts.astype(np.int64, copy=False),
        'num_clusters': int(counts.shape[0]),
        'sample_indices': sample_indices.astype(np.int64, copy=False),
    }
