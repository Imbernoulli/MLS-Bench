"""Score spec for ts-classification."""
from mlsbench.scoring.dsl import *

# accuracy values are fractions [0,1] -> bound=1.0 (not 100.0)

term("accuracy_EthanolConcentration",
    col("accuracy_EthanolConcentration").higher().id()
    .bounded_power(bound=1.0))

term("accuracy_FaceDetection",
    col("accuracy_FaceDetection").higher().id()
    .bounded_power(bound=1.0))

term("accuracy_Handwriting",
    col("accuracy_Handwriting").higher().id()
    .bounded_power(bound=1.0))

setting("EthanolConcentration", weighted_mean(("accuracy_EthanolConcentration", 1.0)))
setting("FaceDetection", weighted_mean(("accuracy_FaceDetection", 1.0)))
setting("Handwriting", weighted_mean(("accuracy_Handwriting", 1.0)))

task(gmean("EthanolConcentration", "FaceDetection", "Handwriting"))
