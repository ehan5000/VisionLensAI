import numpy as np

def generate_dataset(samples: int = 8000, seed: int = 42, out_file: str = "vision_data.npz") -> None:
    """
    Generates a synthetic dataset to model vision quality prediction.

    Features:
      - blur_level:            [0, 1]  (higher = blurrier)
      - focus_error:           [0, 1]  (higher = more out-of-focus)
      - contrast:              [0.5, 1.0] (higher = clearer edges)
      - light_sensitivity:     [0, 1]  (higher = more glare sensitivity)

    Target:
      - vision_quality_score:  [0, 100]
    """
    rng = np.random.default_rng(seed)

    blur = rng.uniform(0.0, 1.0, samples)
    focus_error = rng.uniform(0.0, 1.0, samples)
    contrast = rng.uniform(0.5, 1.0, samples)
    light_sensitivity = rng.uniform(0.0, 1.0, samples)

    noise = rng.normal(0.0, 2.0, samples)

    quality = (
        100.0
        - 42.0 * blur
        - 36.0 * focus_error
        + 20.0 * contrast
        - 18.0 * light_sensitivity
        + noise
    )
    quality = np.clip(quality, 0.0, 100.0)

    X = np.column_stack([blur, focus_error, contrast, light_sensitivity]).astype(np.float32)
    y = quality.reshape(-1, 1).astype(np.float32)

    np.savez(out_file, X=X, y=y)
    print(f"Saved {out_file} | X: {X.shape} | y: {y.shape}")

if __name__ == "__main__":
    generate_dataset()