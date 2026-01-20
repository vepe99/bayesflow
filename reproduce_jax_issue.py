import os
os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import numpy as np
from bayesflow.networks import CompositionalDiffusionModel
import traceback

def test_compositional_diffusion_jax():
    # Set n_compositional to a larger value so we can test the mini-batching branch
    n_datasets = 2
    n_compositional = 10
    num_samples = 4
    dims = 5
    mini_batch_size = 3

    model = CompositionalDiffusionModel()

    z = keras.random.normal((n_datasets, num_samples, dims))
    conditions = keras.random.normal((n_datasets, n_compositional, num_samples, dims))

    def compute_prior_score(xz):
        return -xz # dummy prior score (standard normal)

    # Build model
    model.build(z.shape, conditions.shape)

    print(f"Calling model with mini_batch_size={mini_batch_size} < n_compositional={n_compositional}...")
    try:
        # Pass mini_batch_size through integrate_kwargs
        out = model(z, conditions=conditions, inverse=True, compute_prior_score=compute_prior_score, mini_batch_size=mini_batch_size)
        print("Success!")
        print(f"Output shape: {out.shape}")
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    test_compositional_diffusion_jax()
