# Dropped extractive-QA siblings

`qa-answer-policy` was removed because choosing deliberately invalid output
policies is an anti-gaming demonstration, not a defensible extractive-QA research
axis.  `qa-null-confidence` was removed because applying a sigmoid to the null
margin and thresholding it is a strictly monotone reparameterization of
`qa-null-threshold`; it cannot define an independent scientific question.

Neither task has an active config, harness, solution, parser, or score spec.
