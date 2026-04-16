# Residue Gaussian Cleanup

This note documents the residue-removal techniques we use to clean post-hoc Gaussian part clusters. The main motivation is that boundary Gaussians often receive imperfect supervision from 2D masks, especially near thin structures, occlusion boundaries, or mask leakage regions. As a result, small disconnected Gaussian shards may remain even when the main part cluster is semantically correct.

## Paper-Style Summary

To reduce residual boundary artifacts after part clustering, we apply two complementary post-processing steps. First, we refine feature-space HDBSCAN clusters with a local geometry-aware graph procedure. We preserve the main semantic grouping from HDBSCAN, identify small or low-confidence disconnected components in an xyz-neighborhood graph, and reassign them to neighboring cluster cores only when local contact, feature similarity, and optional SH0 color compatibility jointly support the merge; otherwise they are marked as noise. Second, for interactive editing and export cleanup, we suppress small disconnected connected components that remain after clustering by preserving the largest connected component of each part and hiding components below a fixed residue threshold `tau_r = 1000`. This visual/export cleanup reduces distracting floaters and boundary shards without modifying the learned features or the underlying training objective.

## Implementation Notes

### 1. HDBSCANRefined

`HDBSCANRefined` keeps HDBSCAN as the semantic front-end and adds a local residue refinement stage:

1. Run HDBSCAN in normalized feature space.
2. Assign full-point labels and confidence scores from cluster-center similarity.
3. Build an xyz KNN graph over the Gaussians.
4. Within each initial label, split the cluster into connected components.
5. Treat only non-core components that are either small or low-confidence as residue candidates.
6. Merge a residue candidate into a neighboring cluster core only if the best merge score is sufficiently strong and sufficiently better than the second-best candidate.
7. If the residue is ambiguous, keep it as noise instead of forcing an incorrect merge.

The merge score combines:

- boundary contact ratio on the local graph
- feature compatibility with the target core
- spatial compatibility in xyz
- optional SH0 color compatibility

This makes the refinement local and conservative, which helps preserve part-level consistency while reducing disconnected shards.

### 2. Component-Level Residue Suppression

Even after clustering, some small disconnected components may remain due to noisy boundary supervision. For visualization and active export cleanup, we therefore apply a component-level suppression rule:

1. Decompose each clustered part into xyz connected components.
2. Preserve the largest component.
3. Mark any other component with size `<= tau_r` as residue.
4. Hide these residue components in the GUI.
5. Optionally exclude them from active-part export.

In our implementation, `tau_r = 1000` by default. This cleanup is intentionally **non-destructive**:

- it does not modify feature checkpoints
- it does not retrain the model
- it does not change quantitative evaluation unless explicitly enabled there

Instead, it acts as a pragmatic visualization/export layer for cleaning up boundary floaters and disconnected Gaussian fragments during interactive part editing.

## Korean Summary

경계 부근 Gaussian은 SAM mask supervision이 불안정해서, part clustering 후에도 작은 disconnected shard나 floater가 남을 수 있습니다. 이를 줄이기 위해 우리는 두 가지 후처리 기법을 사용합니다. 첫째, `HDBSCANRefined`는 feature-space HDBSCAN 결과를 유지하면서 xyz graph 위에서 작은/저신뢰 residue component만 주변 core cluster로 보수적으로 merge하거나, 애매하면 noise로 둡니다. 둘째, GUI와 active export 단계에서는 각 part를 connected component로 나눈 뒤 가장 큰 component만 유지하고, 임계값 `tau_r = 1000` 이하의 작은 component는 residue로 간주하여 숨깁니다. 이 단계는 학습 자체를 바꾸지 않는 시각화/내보내기용 cleanup 레이어입니다.

## Scope Note

The suppression layer is designed for **interactive cleanup and export usability**. It should be reported separately from the training loss and separately from the raw clustering method when presenting quantitative evaluation.
