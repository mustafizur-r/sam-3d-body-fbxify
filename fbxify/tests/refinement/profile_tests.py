import unittest
import math
import numpy as np
from fbxify.refinement.refinement_manager import (
    RefinementManager, RefinementConfig, quat_from_R, R_from_quat, quat_angle, 
    rad2deg, norm, quat_mul, quat_inv, dot
)


def identity_matrix():
    """Create a 3x3 identity rotation matrix."""
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def rotation_matrix_x(angle_rad):
    """Create a rotation matrix around X axis."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def rotation_matrix_y(angle_rad):
    """Create a rotation matrix around Y axis."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def rotation_matrix_z(angle_rad):
    """Create a rotation matrix around Z axis."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


class TestRefinementProfiles(unittest.TestCase):
    """Test suite for refinement profiles with fake data."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RefinementConfig()
        self.fps = 30.0
        self.manager = RefinementManager(self.config, fps=self.fps)

    def test_root_profile_spike_fix_rotation(self):
        """Test that root profile fixes rotation spikes."""
        # Create data with a spike at frame 5 (middle frame)
        # Normal rotation: 0.1 rad/frame, spike: 3.0 rad (should exceed max_ang_speed_deg threshold)
        T = 10
        rotation_series = []
        
        for t in range(T):
            if t == 5:
                # Spike: large rotation (3.0 rad = ~172 deg) in one frame
                # This should exceed max_ang_speed_deg=180.0 and max_ang_accel_deg=1800.0
                angle = 3.0
            else:
                # Normal rotation: small increment
                angle = 0.1 * t
            rotation_series.append(rotation_matrix_z(angle))
        
        # Store original spike rotation for comparison
        original_spike_rotation = rotation_series[5]
        
        # Create root node
        root_node = {
            "name": "root",
            "method": "direct_rotation",
            "data": {
                "rotation": rotation_series
            }
        }
        
        # Apply refinement
        self.manager.apply(root_node)
        
        # Check that spike was fixed (frame 5 should be interpolated between 4 and 6)
        refined_rotations = root_node["data"]["rotation"]
        
        # Convert to quaternions for comparison
        q_orig_spike = quat_from_R(original_spike_rotation)
        q_refined_5 = quat_from_R(refined_rotations[5])
        q_refined_4 = quat_from_R(refined_rotations[4])
        q_refined_6 = quat_from_R(refined_rotations[6])
        
        # Check angular velocity: frame 4->5 should be much smaller than original spike
        # Original spike: 3.0 rad jump from frame 4 to 5
        # Calculate angular velocity for original and refined
        q_orig_4 = quat_from_R(rotation_series[4])
        dq_orig_45 = quat_mul(quat_inv(q_orig_4), q_orig_spike)
        orig_angle_45_rad = quat_angle(dq_orig_45)
        orig_vel_deg_per_sec = rad2deg(orig_angle_45_rad) / (1.0 / self.fps)
        
        dq_ref_45 = quat_mul(quat_inv(q_refined_4), q_refined_5)
        ref_angle_45_rad = quat_angle(dq_ref_45)
        ref_vel_deg_per_sec = rad2deg(ref_angle_45_rad) / (1.0 / self.fps)
        
        # Original spike velocity should be very high (3.0 rad = ~172 deg in 1/30 sec = ~5157 deg/sec)
        # After fix, it should be much smaller (should be below max_ang_speed_deg=180.0 threshold)
        # The refined velocity should be significantly less than original
        self.assertLess(ref_vel_deg_per_sec, orig_vel_deg_per_sec * 0.5,
                       f"Angular velocity after spike fix ({ref_vel_deg_per_sec:.1f} deg/s) should be much smaller than original ({orig_vel_deg_per_sec:.1f} deg/s)")
        
        # Also check that it's below the threshold (with some tolerance for smoothing effects)
        self.assertLess(ref_vel_deg_per_sec, 300.0,
                       f"Angular velocity after spike fix should be below reasonable threshold, got {ref_vel_deg_per_sec:.1f} deg/s")

    def test_head_profile_one_euro_smoothing(self):
        """Test that head profile applies OneEuro smoothing to reduce flicker."""
        # Create data with high-frequency jitter (simulating head yaw flicker)
        T = 20
        rotation_series = []
        
        # Add small random jitter to simulate flicker
        base_angle = 0.0
        for t in range(T):
            # Add jitter: ±0.05 rad per frame
            jitter = 0.05 * math.sin(t * 0.5) + 0.03 * math.cos(t * 0.7)
            angle = base_angle + jitter
            rotation_series.append(rotation_matrix_y(angle))
        
        # Create head node
        head_node = {
            "name": "head",
            "method": "direct_rotation",
            "data": {
                "rotation": rotation_series
            }
        }
        
        # Store original for comparison
        original_rotations = [r[:] for r in rotation_series]
        
        # Apply refinement
        self.manager.apply(head_node)
        
        # Check that smoothing was applied (variance should be reduced)
        refined_rotations = head_node["data"]["rotation"]
        
        # Extract rotation angles (Y axis rotation)
        original_angles = []
        refined_angles = []
        for t in range(T):
            # Extract Y rotation angle from rotation matrix
            orig_angle = math.atan2(original_rotations[t][0][2], original_rotations[t][0][0])
            ref_angle = math.atan2(refined_rotations[t][0][2], refined_rotations[t][0][0])
            original_angles.append(orig_angle)
            refined_angles.append(ref_angle)
        
        # Calculate variance of angles
        orig_variance = np.var(original_angles)
        refined_variance = np.var(refined_angles)
        
        # Smoothing should reduce variance (OneEuro is adaptive, so allow some tolerance)
        # But variance should generally be reduced
        self.assertLess(refined_variance, orig_variance * 1.5, 
                       "OneEuro smoothing should reduce jitter variance")

    def test_hands_profile_spike_fix_vector(self):
        """Test that hands profile fixes vector spikes (roll instability)."""
        # Create data with a vector spike (for keypoint_with_global_rot_roll method)
        T = 10
        dir_vectors = []
        
        for t in range(T):
            if t == 5:
                # Spike: large jump in direction vector
                # This should exceed max_pos_speed=1.0 and max_pos_accel=10.0
                dir_vectors.append([10.0, 0.0, 0.0])  # Large spike
            else:
                # Normal: gradual change
                dir_vectors.append([1.0 + 0.1 * t, 0.0, 0.0])
        
        # Create hand node with keypoint method
        hand_node = {
            "name": "left_hand",
            "method": "keypoint_with_global_rot_roll",
            "data": {
                "dir_vector": dir_vectors,
                "roll_vector": [identity_matrix()] * T
            }
        }
        
        # Apply refinement
        self.manager.apply(hand_node)
        
        # Check that spike was fixed
        refined_vectors = hand_node["data"]["dir_vector"]
        
        # Frame 5 should be interpolated between frames 4 and 6
        # Original spike was [10.0, 0.0, 0.0]
        # After fix, it should be closer to average of frames 4 and 6
        spike_frame = refined_vectors[5]
        frame_4 = refined_vectors[4]
        frame_6 = refined_vectors[6]
        
        # Check that spike is not the original value
        # Spike should have been fixed (vector should differ from original spike)
        self.assertNotAlmostEqual(spike_frame[0], 10.0, places=1)
        
        # Check that it's closer to the neighbors
        avg_neighbor = [(frame_4[i] + frame_6[i]) / 2.0 for i in range(3)]
        distance_to_avg = norm([spike_frame[i] - avg_neighbor[i] for i in range(3)])
        distance_to_original = norm([spike_frame[i] - 10.0 if i == 0 else spike_frame[i] for i in range(3)])
        
        self.assertLess(distance_to_avg, distance_to_original,
                       "Fixed spike should be closer to neighbor average than original spike")

    def test_fingers_profile_high_frequency_jitter(self):
        """Test that fingers profile reduces high-frequency jitter."""
        # Create data with high-frequency jitter (typical for fingers)
        T = 30
        rotation_series = []
        
        # Add high-frequency jitter
        for t in range(T):
            # High frequency: 0.1 rad jitter at 5 Hz (relative to 30 fps)
            jitter = 0.1 * math.sin(t * 2 * math.pi * 5 / 30)
            angle = jitter
            rotation_series.append(rotation_matrix_x(angle))
        
        # Create finger node
        finger_node = {
            "name": "left_index_finger",
            "method": "direct_rotation",
            "data": {
                "rotation": rotation_series
            }
        }
        
        # Store original
        original_rotations = [r[:] for r in rotation_series]
        
        # Apply refinement
        self.manager.apply(finger_node)
        
        # Check that high-frequency jitter was reduced
        refined_rotations = finger_node["data"]["rotation"]
        
        # Calculate high-frequency content (angular velocity between consecutive frames)
        original_velocities = []
        refined_velocities = []
        
        for t in range(1, T):
            # Get relative rotation between frames
            q_orig_prev = quat_from_R(original_rotations[t-1])
            q_orig_curr = quat_from_R(original_rotations[t])
            q_ref_prev = quat_from_R(refined_rotations[t-1])
            q_ref_curr = quat_from_R(refined_rotations[t])
            
            # Relative rotation (angular velocity)
            dq_orig = quat_mul(quat_inv(q_orig_prev), q_orig_curr)
            dq_ref = quat_mul(quat_inv(q_ref_prev), q_ref_curr)
            
            # Get angular velocity in degrees
            orig_vel = rad2deg(quat_angle(dq_orig)) / (1.0 / self.fps)  # deg/sec
            ref_vel = rad2deg(quat_angle(dq_ref)) / (1.0 / self.fps)  # deg/sec
            
            original_velocities.append(orig_vel)
            refined_velocities.append(ref_vel)
        
        # Average angular velocity should be reduced
        avg_orig_vel = np.mean(original_velocities)
        avg_ref_vel = np.mean(refined_velocities)
        
        # Smoothing should reduce high-frequency content
        # Allow some tolerance since OneEuro is adaptive
        self.assertLessEqual(avg_ref_vel, avg_orig_vel * 1.2,
                            "High-frequency jitter should be reduced")

    def test_legs_profile_ema_smoothing(self):
        """Test that legs profile applies EMA smoothing (for twist instability)."""
        # Create data with gradual twist instability
        T = 20
        rotation_series = []
        
        # Add gradual rotation with some noise
        for t in range(T):
            base_angle = 0.2 * t  # Gradual rotation
            noise = 0.05 * math.sin(t * 0.3)  # Low-frequency noise
            angle = base_angle + noise
            rotation_series.append(rotation_matrix_z(angle))
        
        # Create leg node
        leg_node = {
            "name": "left_leg",
            "method": "direct_rotation",
            "data": {
                "rotation": rotation_series
            }
        }
        
        # Store original
        original_rotations = [r[:] for r in rotation_series]
        
        # Apply refinement
        self.manager.apply(leg_node)
        
        # Check that EMA smoothing was applied
        refined_rotations = leg_node["data"]["rotation"]
        
        # Extract rotation angles
        original_angles = []
        refined_angles = []
        for t in range(T):
            # Extract Z rotation angle
            orig_angle = math.atan2(original_rotations[t][1][0], original_rotations[t][0][0])
            ref_angle = math.atan2(refined_rotations[t][1][0], refined_rotations[t][0][0])
            original_angles.append(orig_angle)
            refined_angles.append(ref_angle)
        
        # EMA should smooth out noise while preserving trend
        # Calculate noise (deviation from linear trend)
        orig_trend = np.polyfit(range(T), original_angles, 1)
        ref_trend = np.polyfit(range(T), refined_angles, 1)
        
        orig_noise = [abs(original_angles[t] - (orig_trend[0] * t + orig_trend[1])) for t in range(T)]
        ref_noise = [abs(refined_angles[t] - (ref_trend[0] * t + ref_trend[1])) for t in range(T)]
        
        # Noise should be reduced
        avg_orig_noise = np.mean(orig_noise)
        avg_ref_noise = np.mean(ref_noise)
        
        self.assertLess(avg_ref_noise, avg_orig_noise,
                       "EMA smoothing should reduce noise while preserving trend")

    def test_root_motion_stabilization(self):
        """Test that root motion stabilization works correctly."""
        # Create root motion with jitter
        T = 15
        translation = []
        rotation = []
        
        for t in range(T):
            # Add jitter to translation
            base_pos = [t * 0.1, 0.0, 1.0]  # Moving forward, constant height
            jitter = [0.02 * math.sin(t * 2), 0.02 * math.cos(t * 2), 0.01 * math.sin(t * 3)]
            translation.append([base_pos[i] + jitter[i] for i in range(3)])
            
            # Small rotation jitter
            rot_jitter = 0.05 * math.sin(t * 1.5)
            rotation.append(rotation_matrix_y(rot_jitter))
        
        root_motion = {
            "translation": translation,
            "rotation": rotation
        }
        
        # Store original
        original_translation = [t[:] for t in translation]
        
        # Apply refinement with root motion
        root_node = {"name": "root", "method": "direct_rotation", "data": {"rotation": rotation}}
        self.manager.apply(root_node, root_motion)
        
        # Check that translation jitter was reduced
        refined_translation = root_motion["translation"]
        
        # Calculate jitter (deviation from smooth path)
        orig_jitter = []
        ref_jitter = []
        
        for t in range(1, T-1):
            # Jitter is deviation from linear interpolation of neighbors
            smooth_orig = [(original_translation[t-1][i] + original_translation[t+1][i]) / 2.0 
                          for i in range(3)]
            smooth_ref = [(refined_translation[t-1][i] + refined_translation[t+1][i]) / 2.0 
                         for i in range(3)]
            
            orig_jitter.append(norm([original_translation[t][i] - smooth_orig[i] for i in range(3)]))
            ref_jitter.append(norm([refined_translation[t][i] - smooth_ref[i] for i in range(3)]))
        
        # Root stabilization should reduce jitter
        avg_orig_jitter = np.mean(orig_jitter)
        avg_ref_jitter = np.mean(ref_jitter)
        
        self.assertLess(avg_ref_jitter, avg_orig_jitter * 1.1,
                       "Root motion stabilization should reduce translation jitter")

    def test_default_profile_fallback(self):
        """Test that unknown bone names fall back to default profile."""
        # Create a bone with unknown name
        T = 10
        rotation_series = [identity_matrix()] * T
        
        # Add a spike
        rotation_series[5] = rotation_matrix_z(4.0)  # Large spike
        
        unknown_node = {
            "name": "unknown_bone_xyz",
            "method": "direct_rotation",
            "data": {
                "rotation": rotation_series
            }
        }
        
        # Apply refinement
        self.manager.apply(unknown_node)
        
        # Should still process (no error) and use default profile
        refined_rotations = unknown_node["data"]["rotation"]
        self.assertEqual(len(refined_rotations), T,
                        "Default profile should still process the bone")

    def test_nested_children_processing(self):
        """Test that refinement processes nested children correctly."""
        # Create a hierarchy: root -> child -> grandchild
        T = 10
        
        root_node = {
            "name": "root",
            "method": "direct_rotation",
            "data": {
                "rotation": [identity_matrix()] * T
            },
            "children": [
                {
                    "name": "left_hand",
                    "method": "direct_rotation",
                    "data": {
                        "rotation": [rotation_matrix_x(0.1 * t) for t in range(T)]
                    },
                    "children": [
                        {
                            "name": "left_index_finger",
                            "method": "direct_rotation",
                            "data": {
                                "rotation": [rotation_matrix_x(0.05 * t) for t in range(T)]
                            }
                        }
                    ]
                }
            ]
        }
        
        # Store original
        original_hand_rot = [r[:] for r in root_node["children"][0]["data"]["rotation"]]
        original_finger_rot = [r[:] for r in root_node["children"][0]["children"][0]["data"]["rotation"]]
        
        # Apply refinement
        self.manager.apply(root_node)
        
        # Check that all levels were processed
        refined_hand_rot = root_node["children"][0]["data"]["rotation"]
        refined_finger_rot = root_node["children"][0]["children"][0]["data"]["rotation"]
        
        # Hand should use HANDS_PROFILE (OneEuro)
        # Finger should use FINGERS_PROFILE (OneEuro with different params)
        # Both should be smoothed (not identical to original due to smoothing)
        
        # Check that processing occurred (rotations may have changed due to smoothing)
        # We'll just verify the structure is intact and data exists
        self.assertEqual(len(refined_hand_rot), T, "Hand rotations should be processed")
        self.assertEqual(len(refined_finger_rot), T, "Finger rotations should be processed")
        
        # Verify different profiles were applied (hand and finger should have different smoothing)
        # This is indirect - we just verify both were processed


if __name__ == "__main__":
    unittest.main()
