"""Literal covariance-and-mean plan for the trusted ExactGP builder."""


def surface_config():
    return {"kernel": "rbf", "ard": False, "mean": "constant"}
