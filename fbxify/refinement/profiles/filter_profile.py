from dataclasses import dataclass

@dataclass
class FilterProfile:
    """
    Defines filtering behavior for a semantic bone group.
    """
    # --- spike thresholds ---
    max_pos_speed: float = 3.0        # units / sec
    max_pos_accel: float = 30.0       # units / sec^2
    max_ang_speed_deg: float = 720.0  # deg / sec
    max_ang_accel_deg: float = 7200.0 # deg / sec^2

    # --- smoothing ---
    method: str = "one_euro"          # "ema" | "butterworth" | "one_euro"
    cutoff_hz: float = 4.0
    one_euro_min_cutoff: float = 1.5
    one_euro_beta: float = 0.5
    one_euro_d_cutoff: float = 1.0

    # --- root-only tuning ---
    root_cutoff_xy_hz: float = 2.0
    root_cutoff_z_hz: float = 0.8

