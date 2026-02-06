from fbxify.refinement.profiles.filter_profile import FilterProfile

FINGERS_PROFILE = FilterProfile( # fingers typically has high-frequency jitter
    max_pos_speed=0.5,
    max_pos_accel=5.0,
    max_ang_speed_deg=180.0,
    max_ang_accel_deg=1800.0,
    method="one_euro",
    one_euro_min_cutoff=0.8,
    one_euro_beta=0.2,
)
