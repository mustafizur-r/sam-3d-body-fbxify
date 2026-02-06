from fbxify.refinement.profiles.filter_profile import FilterProfile

LEGS_PROFILE = FilterProfile( # legs typically has twist instability
    max_pos_speed=2.0,
    max_pos_accel=25.0,
    max_ang_speed_deg=400.0,
    max_ang_accel_deg=4000.0,
    method="ema",
    cutoff_hz=3.0,
)
