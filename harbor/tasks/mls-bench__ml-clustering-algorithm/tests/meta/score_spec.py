"""Score spec for ml-clustering-algorithm."""
from mlsbench.scoring.dsl import *

# ari, nmi: 0-1 scale, higher is better
# silhouette: -1 to 1 scale, higher is better
# varied_density metrics removed — no corresponding test_cmd label in config

term("ari_blobs",
    col("ari_blobs").higher().id()
    .bounded_power(bound=1.0))

term("nmi_blobs",
    col("nmi_blobs").higher().id()
    .bounded_power(bound=1.0))

term("silhouette_blobs",
    col("silhouette_blobs").higher().id()
    .bounded_power(bound=1.0))

term("ari_moons",
    col("ari_moons").higher().id()
    .bounded_power(bound=1.0))

term("nmi_moons",
    col("nmi_moons").higher().id()
    .bounded_power(bound=1.0))

term("silhouette_moons",
    col("silhouette_moons").higher().id()
    .bounded_power(bound=1.0))

term("ari_digits",
    col("ari_digits").higher().id()
    .bounded_power(bound=1.0))

term("nmi_digits",
    col("nmi_digits").higher().id()
    .bounded_power(bound=1.0))

term("silhouette_digits",
    col("silhouette_digits").higher().id()
    .bounded_power(bound=1.0))

setting("blobs", weighted_mean(("ari_blobs", 1.0), ("nmi_blobs", 1.0), ("silhouette_blobs", 1.0)))
setting("moons", weighted_mean(("ari_moons", 1.0), ("nmi_moons", 1.0), ("silhouette_moons", 1.0)))
setting("digits", weighted_mean(("ari_digits", 1.0), ("nmi_digits", 1.0), ("silhouette_digits", 1.0)))

task(gmean("blobs", "moons", "digits"))
