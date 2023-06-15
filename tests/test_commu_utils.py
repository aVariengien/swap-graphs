

from swap_graphs.communities_utils import (
    create_random_communities,
    average_cluster_size,
    average_class_entropy
)

from swap_graphs.core import (
    ActivationStore,
    CompMetric,
    ModelComponent,
    SwapGraph,
    WildPosition,
    find_important_components,
    SgraphDataset,
    compute_clustering_metrics,
)


## Test the random communities and the entropy and cluster size computation

def test_commu_utils():

    test_list_components = [
        ModelComponent(position=42, layer=l, name="z", head=h, position_label="test")
        for l in range(12)
        for h in range(12)
    ]

    random_comus = create_random_communities(
        test_list_components, n_samples=100, n_classes=20
    )
    assert (
        average_cluster_size(random_comus, 100) >= 0.04
        and average_cluster_size(random_comus, 100) <= 0.06
    )

    random_comus = create_random_communities(
        test_list_components, n_samples=100, n_classes=5
    )

    assert (
        average_cluster_size(random_comus, 100) >= 0.18
        and average_cluster_size(random_comus, 100) <= 0.22
    )

    random_comus = create_random_communities(
        test_list_components, n_samples=100, n_classes=1
    )  # all samples in the same class
    assert (
        abs(average_class_entropy(random_comus) - 524.76499) < 1e-3
    )  #  524.76499 = log(100!)

    random_comus = create_random_communities(
        test_list_components, n_samples=100, n_classes=int(1e9)
    )  # in the limit, no collision, one sample per classes, the partitions are made of singletons.

    assert average_class_entropy(random_comus) < 1e-5
