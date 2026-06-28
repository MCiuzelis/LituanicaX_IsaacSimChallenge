"""MuSHR nano v2 — cone-track navigation.  Obs: 23 (speed, yaw, centerline error,
heading err, corner dist/curvature, 5 lookahead points with body-frame x/y/curvature).
Action: 2 (throttle ∈ [-1,1], steering ∈ [-1,1] → ±max_steer_rad)."""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

# ── Assets ─────────────────────────────────────────────────────────────────
_ASSETS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../assets")
)

MUSHR_USD = os.path.join(_ASSETS_DIR, "mushr_nano_v2.usd")
TRACK_USD = os.path.join(_ASSETS_DIR, "Track.usdc")
_TRACKPOINTS_CSV = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../TrackPoints.csv")
)
WALL_USD = os.path.join(_ASSETS_DIR, "Walls.usdc")

# ---------------------------------------------------------------------------
# Physical constants — MuSHR nano v2 (fixed by the USD / hardware)
# ---------------------------------------------------------------------------
WHEEL_RADIUS = 0.037  # m  (82 mm diameter wheels)
TRACK_SCALE = 0.85


def _extract_wall_segments_xy() -> list:
    """Extract 2D wall segments from the USD Walls mesh for collision detection."""
    try:
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        wall_root = stage.GetPrimAtPath("/World/Walls")
        if not wall_root.IsValid():
            return []

        xf_cache = UsdGeom.XformCache()
        segments: list = []

        for prim in Usd.PrimRange(wall_root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points_attr = mesh.GetPointsAttr().Get()
            if points_attr is None:
                continue
            counts_attr = mesh.GetFaceVertexCountsAttr().Get()
            indices_attr = mesh.GetFaceVertexIndicesAttr().Get()
            if counts_attr is None or indices_attr is None:
                continue

            world_xf = xf_cache.GetLocalToWorldTransform(prim)
            world_pts = [
                world_xf.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                for p in points_attr
            ]

            idx_ptr = 0
            for fc in counts_attr:
                face = [int(indices_attr[idx_ptr + i]) for i in range(fc)]
                idx_ptr += fc
                for k in range(1, fc - 1):
                    tri = (face[0], face[k], face[k + 1])
                    pts3 = (world_pts[tri[0]], world_pts[tri[1]], world_pts[tri[2]])
                    for e in range(3):
                        a3 = pts3[e]
                        b3 = pts3[(e + 1) % 3]
                        if abs(a3[2] - b3[2]) > 0.05:
                            continue
                        dx = b3[0] - a3[0]
                        dy = b3[1] - a3[1]
                        if dx * dx + dy * dy < 1e-4:
                            continue
                        segments.append(
                            [
                                [float(a3[0]), float(a3[1])],
                                [float(b3[0]), float(b3[1])],
                            ]
                        )
        return segments
    except Exception as e:
        print(f"[ConeTrackEnv] Wall segment extraction failed: {e}")
        return []


# ── Config ─────────────────────────────────────────────────────────────────


@configclass
class ConeTrackEnvCfg(DirectRLEnvCfg):
    """Single source of truth for every number you might want to change."""

    # ── Core env parameters ────────────────────────────────────────────────
    decimation: int = 8  # policy at ~60 Hz  (sim 480 Hz / 8)
    episode_length_s: float = 90.0  # 180 s × 60 Hz = 10 800 steps
    action_space: int = 2
    observation_space: int = (
        23  # Expanded from 8 to 23 to capture lookahead spatial points
    )
    state_space: int = 0
    ui_window_class_type = None

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 480.0, render_interval=8)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=200,
        env_spacing=0.0,
        replicate_physics=True,
    )

    # ── Track observation ──────────────────────────────────────────────────
    track_max_dist_m: float = 0.3  # normaliser for distance to centerline
    corner_curvature_threshold: float = (
        0.5  # min curvature (1/m) to count as a corner peak
    )
    corner_lookahead_max_m: float = (
        5.0  # normaliser for distance-to-next-corner and lookahead coords
    )
    max_curvature: float = 10.0  # normaliser for curvature (1/m)

    # ── Spawn ──────────────────────────────────────────────────────────────
    spawn_grace_steps: int = 8

    preset_spawn_points: list = [
        (0.756, 5.0, 20.8),
        (3.91, 3.31, -156.0),
        (3.62, 1.97, -154.48),
        (-5.46, 3.7, 0.0),
        (-3.6, 4.7, -217.86),
    ]
    spawn_x_scale: float = -1.15 * TRACK_SCALE
    spawn_y_scale: float = -1.15 * TRACK_SCALE

    duplicate_with_flipped_yaw: bool = True
    spawn_yaw_jitter_rad: float = 0.1

    # ── Dynamics ───────────────────────────────────────────────────────────
    car_mass_kg: float = 1.27
    max_velocity_m_s: float = 6.7
    drive_torque: float = 0.068
    brake_torque_base: float = 0.028
    brake_torque_gain: float = 0.310
    brake_throttle_scale: float = 0.4
    brake_release_omega: float = 2.0
    ground_static_friction: float = 2.0
    ground_dynamic_friction: float = 1.6
    suspension_stiffness: float = 1e6
    max_steer_rad: float = 0.488

    # ── Termination ────────────────────────────────────────────────────────
    wall_collision_radius_m: float = 0.15
    slow_stop_speed_fraction: float = 0.04

    # ── Rewards ────────────────────────────────────────────────────────────
    alive_weight: float = 0.02
    distance_weight: float = 4.0
    steer_deadzone_weight: float = 0.003
    steer_deadzone: float = 0.05
    steer_rate_weight: float = 0.003
    throttle_rate_weight: float = 0.002
    slip_weight: float = 0.03

    roll_weight: float = 0.1

    # ── Hardware (articulation) ────────────────────────────────────────────
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=MUSHR_USD,
            scale=(0.755, 0.755, 0.755),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=10000.0,
                max_angular_velocity=100000.0,
                max_depenetration_velocity=1.0,
                max_contact_impulse=5.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            joint_pos={
                "back_left_wheel_throttle": 0.0,
                "back_right_wheel_throttle": 0.0,
                "front_left_wheel_throttle": 0.0,
                "front_right_wheel_throttle": 0.0,
                "front_left_wheel_steer": 0.0,
                "front_right_wheel_steer": 0.0,
                "front_left_wheel_suspension": 0.0,
                "front_right_wheel_suspension": 0.0,
                "back_left_wheel_suspension": 0.0,
                "back_right_wheel_suspension": 0.0,
            },
        ),
        actuators={
            "steering": ImplicitActuatorCfg(
                joint_names_expr=["front_left_wheel_steer", "front_right_wheel_steer"],
                stiffness=100.0,
                damping=10.0,
                velocity_limit_sim=6.51,
                effort_limit_sim=3.2,
            ),
        },
    )

    def __post_init__(self):
        self.robot_cfg.actuators["throttle"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_throttle"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=5.0,
            velocity_limit_sim=1e9,
        )
        self.robot_cfg.actuators["suspension"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_suspension"],
            stiffness=self.suspension_stiffness,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1e9,
            velocity_limit_sim=1e9,
        )


# ── Env ────────────────────────────────────────────────────────────────────


class ConeTrackEnv(DirectRLEnv):
    cfg: ConeTrackEnvCfg

    def __init__(self, cfg: ConeTrackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint indices
        self._rear_throttle_ids, _ = self.robot.find_joints(
            ["back_left_wheel_throttle", "back_right_wheel_throttle"]
        )
        self._front_throttle_ids, _ = self.robot.find_joints(
            ["front_left_wheel_throttle", "front_right_wheel_throttle"]
        )
        self._steer_ids, _ = self.robot.find_joints(
            ["front_left_wheel_steer", "front_right_wheel_steer"]
        )

        self._steer_tan = torch.zeros(self.num_envs, device=self.device)
        self._all_throttle_ids = torch.tensor(
            self._rear_throttle_ids + self._front_throttle_ids,
            dtype=torch.long,
            device=self.device,
        )

        self._wheel_torque = torch.zeros(self.num_envs, 4, device=self.device)

        # Correct the total mass
        _pv = self.robot.root_physx_view
        _masses = _pv.get_masses()
        _native_total = _masses[0].sum().item()
        _mass_scale = cfg.car_mass_kg / _native_total
        _env_ids_cpu = torch.arange(self.num_envs, dtype=torch.int, device="cpu")
        _pv.set_masses(_masses * _mass_scale, _env_ids_cpu)
        _pv.set_inertias(_pv.get_inertias() * _mass_scale, _env_ids_cpu)

        self._filtered_steer = torch.zeros(self.num_envs, device=self.device)

        self._steer_cmd_curr = torch.zeros(self.num_envs, device=self.device)
        self._steer_cmd_prev = torch.zeros(self.num_envs, device=self.device)
        self._throttle_cmd_curr = torch.zeros(self.num_envs, device=self.device)
        self._throttle_cmd_prev = torch.zeros(self.num_envs, device=self.device)

        self._wall_ever_touched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self._is_flipped_spawn = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._direction_ema_alpha = 0.005
        self._mean_ep_len_normal = torch.tensor(100.0, device=self.device)
        self._mean_ep_len_flipped = torch.tensor(100.0, device=self.device)

        _wall_segs = _extract_wall_segments_xy()
        if _wall_segs:
            import numpy as _np

            _seg_arr = torch.from_numpy(_np.asarray(_wall_segs, dtype=_np.float32)).to(
                self.device
            )
            self._wall_seg_a = _seg_arr[:, 0, :].contiguous()
            self._wall_seg_b = _seg_arr[:, 1, :].contiguous()
        else:
            self._wall_seg_a = None
            self._wall_seg_b = None

        self._centerline_data = self._load_centerline()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        track_cfg = sim_utils.UsdFileCfg(
            usd_path=TRACK_USD, scale=(TRACK_SCALE, TRACK_SCALE, TRACK_SCALE)
        )
        track_cfg.func(
            "/World/Track",
            track_cfg,
            translation=(0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )

        wall_cfg = sim_utils.UsdFileCfg(
            usd_path=WALL_USD, scale=(TRACK_SCALE, TRACK_SCALE, TRACK_SCALE)
        )
        wall_cfg.func(
            "/World/Walls",
            wall_cfg,
            translation=(0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )

        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        stage = omni.usd.get_context().get_stage()

        wheel_link_names = {
            "front_left_wheel_link",
            "front_right_wheel_link",
            "back_left_wheel_link",
            "back_right_wheel_link",
        }
        wheel_root = stage.GetPrimAtPath("/World/envs/env_0/Robot/mushr_nano")
        if wheel_root.IsValid():
            for prim in Usd.PrimRange(wheel_root):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                if prim.GetParent().GetName() not in wheel_link_names:
                    continue
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision.GetApproximationAttr().Set("convexHull")

        for wall_path, collision_enabled in (("/World/Walls", True),):
            wall_root = stage.GetPrimAtPath(wall_path)
            if not wall_root.IsValid():
                continue
            for prim in Usd.PrimRange(wall_root):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                wall_collision = UsdPhysics.CollisionAPI.Apply(prim)
                wall_collision.GetCollisionEnabledAttr().Set(collision_enabled)
                wall_mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                wall_mesh_collision.GetApproximationAttr().Set("none")
            wall_rb = UsdPhysics.RigidBodyAPI.Apply(wall_root)
            wall_rb.GetRigidBodyEnabledAttr().Set(True)
            wall_rb.GetKinematicEnabledAttr().Set(True)
            UsdGeom.Imageable(wall_root).MakeInvisible()

        suspension_joint_names = (
            "front_left_wheel_suspension",
            "front_right_wheel_suspension",
            "back_left_wheel_suspension",
            "back_right_wheel_suspension",
        )
        for env_id in range(self.num_envs):
            base_link_path = f"/World/envs/env_{env_id}/Robot/mushr_nano/base_link"
            for joint_name in suspension_joint_names:
                joint_prim = stage.GetPrimAtPath(f"{base_link_path}/{joint_name}")
                if not joint_prim.IsValid():
                    continue
                for prop in list(joint_prim.GetProperties()):
                    prop_name = prop.GetName()
                    if "stiffness" in prop_name or "damping" in prop_name:
                        attr = joint_prim.GetAttribute(prop_name)
                        if attr.IsValid():
                            attr.Block()

        ground_cfg = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=self.cfg.ground_static_friction,
                dynamic_friction=self.cfg.ground_dynamic_friction,
                restitution=0.05,
                friction_combine_mode="multiply",
                restitution_combine_mode="min",
            )
        )
        ground_cfg.func("/World/GroundPlane", ground_cfg)

        rubber_mat_path = "/World/PhysicsMaterials/WheelRubber"
        rubber_mat_prim = stage.DefinePrim(rubber_mat_path, "Material")
        UsdPhysics.MaterialAPI.Apply(rubber_mat_prim)
        phys_rubber = UsdPhysics.MaterialAPI(rubber_mat_prim)
        phys_rubber.CreateStaticFrictionAttr().Set(1.0)
        phys_rubber.CreateDynamicFrictionAttr().Set(1.0)
        phys_rubber.CreateRestitutionAttr().Set(0.05)

        self.scene.clone_environments(copy_from_source=False)

        rubber_mat = UsdShade.Material(rubber_mat_prim)
        for env_id in range(self.num_envs):
            wheel_root_env = stage.GetPrimAtPath(
                f"/World/envs/env_{env_id}/Robot/mushr_nano"
            )
            if not wheel_root_env.IsValid():
                continue
            for prim in Usd.PrimRange(wheel_root_env):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                if prim.GetParent().GetName() not in wheel_link_names:
                    continue
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    rubber_mat, UsdShade.Tokens.strongerThanDescendants, "physics"
                )

        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_wall_hit(self) -> torch.Tensor:
        if self._wall_seg_a is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        pos_xy = self.robot.data.root_pos_w[:, :2].unsqueeze(1)
        a = self._wall_seg_a.unsqueeze(0)
        b = self._wall_seg_b.unsqueeze(0)
        ab = b - a
        ap = pos_xy - a
        t = (ap * ab).sum(dim=-1) / (ab * ab).sum(dim=-1).clamp(min=1e-8)
        t = t.clamp(0.0, 1.0)
        closest = a + t.unsqueeze(-1) * ab
        dist_sq = (pos_xy - closest).pow(2).sum(dim=-1)
        return dist_sq.min(dim=1).values.sqrt() < self.cfg.wall_collision_radius_m

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        raw = actions[:, :2].clamp(-1.0, 1.0)
        u = raw[:, 0].unsqueeze(-1)
        wheel_omega = self.robot.data.joint_vel[:, self._all_throttle_ids].clamp(
            min=0.0
        )

        omega_noload = u.clamp(min=1e-3) * self.cfg.max_velocity_m_s / WHEEL_RADIUS
        drive_rolloff = (1.0 - wheel_omega / omega_noload).clamp(0.0, 1.0)
        drive_torque = self.cfg.drive_torque * drive_rolloff

        brake_level = (-u).clamp(min=0.0) * self.cfg.brake_throttle_scale
        brake_fade = (wheel_omega / self.cfg.brake_release_omega).clamp(0.0, 1.0)
        brake_torque = (
            -(self.cfg.brake_torque_base + self.cfg.brake_torque_gain * brake_level)
            * brake_fade
        )

        self._wheel_torque = torch.where(
            u > 0.0,
            drive_torque,
            torch.where(u < 0.0, brake_torque, torch.zeros_like(brake_torque)),
        )

        self._filtered_steer = raw[:, 1]
        steer_ang = self._filtered_steer * self.cfg.max_steer_rad
        self._steer_tan = torch.tan(steer_ang)

        self._steer_cmd_prev = self._steer_cmd_curr
        self._steer_cmd_curr = raw[:, 1]
        self._throttle_cmd_prev = self._throttle_cmd_curr
        self._throttle_cmd_curr = raw[:, 0]

    def _apply_action(self) -> None:
        wall_hit_now = self._compute_wall_hit()
        in_grace = self.episode_length_buf < self.cfg.spawn_grace_steps
        self._wall_ever_touched |= wall_hit_now & ~in_grace

        wall_mask = self._wall_ever_touched
        wheel_torque = self._wheel_torque.clone()
        wheel_torque[wall_mask] = 0.0
        self.robot.set_joint_effort_target(
            wheel_torque, joint_ids=self._all_throttle_ids
        )

        steer_positions = self._steer_tan.unsqueeze(-1).expand(-1, 2).clone()
        steer_positions[wall_mask] = 0.0
        self.robot.set_joint_position_target(steer_positions, joint_ids=self._steer_ids)

    def _load_centerline(self) -> dict:
        if not os.path.isfile(_TRACKPOINTS_CSV):
            return None
        try:
            pts = []
            with open(_TRACKPOINTS_CSV, newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:
                        continue
                    x_blender = float(row[0])
                    y_blender = float(row[1])
                    x_isaac = x_blender * self.cfg.spawn_x_scale
                    y_isaac = y_blender * self.cfg.spawn_y_scale
                    pts.append([x_isaac, y_isaac])
            if not pts:
                return None
            centerline = torch.tensor(pts, dtype=torch.float32, device=self.device)
            M = centerline.shape[0]

            prev = torch.cat([centerline[-1:], centerline[:-1]], dim=0)
            nxt = torch.cat([centerline[1:], centerline[:1]], dim=0)

            v1 = centerline - prev
            v2 = nxt - centerline
            v3 = nxt - prev

            n1 = torch.norm(v1, dim=1)
            n2 = torch.norm(v2, dim=1)
            n3 = torch.norm(v3, dim=1)

            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            curvature = 2.0 * cross.abs() / (n1 * n2 * n3).clamp(min=1e-8)

            tangents = v3 / n3.unsqueeze(-1).clamp(min=1e-8)

            arc_length = torch.zeros(M, device=self.device)
            arc_length[1:] = torch.cumsum(n1[1:], dim=0)
            track_length = arc_length[-1] + n1[0]

            left = torch.cat([curvature[-1:], curvature[:-1]], dim=0)
            right = torch.cat([curvature[1:], curvature[:1]], dim=0)
            is_local_max = (curvature > left) & (curvature >= right)
            is_corner = is_local_max & (curvature > self.cfg.corner_curvature_threshold)
            corner_idx = torch.where(is_corner)[0]

            print(
                f"[ConeTrackEnv] Centerline: {M} points, "
                f"track length {track_length:.2f} m, "
                f"{len(corner_idx)} corners detected"
            )
            return {
                "centerline": centerline,
                "tangents": tangents,
                "curvature": curvature,
                "arc_length": arc_length,
                "track_length": track_length,
                "corner_idx": corner_idx,
                "num_points": M,
            }
        except Exception as e:
            print(f"[ConeTrackEnv] Failed to load centerline: {e}")
            return None

    def _get_observations(self) -> dict:
        cfg = self.cfg
        N = self.num_envs

        # 1) Wheel velocity
        wheel_omega_mean = (
            self.robot.data.joint_vel[:, self._all_throttle_ids].abs().mean(dim=-1)
        )
        wheel_vel_norm = (wheel_omega_mean * WHEEL_RADIUS / cfg.max_velocity_m_s).clamp(
            0.0, 1.0
        )

        # 2) Forward car velocity
        car_vel_fwd = self.robot.data.root_lin_vel_b[:, 0]
        car_vel_fwd_norm = (car_vel_fwd / cfg.max_velocity_m_s).clamp(0.0, 1.0)

        # 3) Lateral car velocity
        lateral_vel = self.robot.data.root_lin_vel_b[:, 1]
        lateral_vel_norm = (lateral_vel / cfg.max_velocity_m_s).clamp(-1.0, 1.0)

        # 4) Yaw rate
        yaw_rate = self.robot.data.root_ang_vel_b[:, 2]
        yaw_rate_norm = (yaw_rate / 10.0).clamp(-1.0, 1.0)

        pos_xy = self.robot.data.root_pos_w[:, :2]

        # 5-22) Track-geometry observations
        if self._centerline_data is not None:
            cl = self._centerline_data["centerline"]
            tangents = self._centerline_data["tangents"]
            curvature = self._centerline_data["curvature"]
            arc_len = self._centerline_data["arc_length"]
            track_len = self._centerline_data["track_length"]
            corner_idx = self._centerline_data["corner_idx"]
            M = self._centerline_data["num_points"]

            # Nearest point on centerline
            diff = pos_xy.unsqueeze(1) - cl.unsqueeze(0)
            sq_dist = (diff * diff).sum(dim=-1)
            nearest_d = sq_dist.min(dim=1)
            nearest_idx = nearest_d.indices

            # 5) Distance to centerline (Crosstrack Error)
            dist_cl = nearest_d.values.sqrt()
            dist_cl_norm = (dist_cl / cfg.track_max_dist_m).clamp(0.0, 1.0)

            # 6) Relative heading
            q = self.robot.data.root_quat_w
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            car_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            car_fwd = torch.stack([torch.cos(car_yaw), torch.sin(car_yaw)], dim=-1)

            tgt = tangents[nearest_idx]
            vel_w = self.robot.data.root_lin_vel_w[:, :2]
            fwd_dir = ((vel_w * tgt).sum(dim=-1) > 0.0).float()
            idx_sign = (fwd_dir * 2 - 1).long()

            dot = (car_fwd * tgt).sum(dim=-1).clamp(-1.0, 1.0)
            heading_err = torch.acos(dot.abs())

            cross_val = car_fwd[:, 0] * tgt[:, 1] - car_fwd[:, 1] * tgt[:, 0]
            rel_heading = heading_err * torch.sign(cross_val)
            rel_heading_norm = (rel_heading / math.pi).clamp(-1.0, 1.0)

            # 7-8) Next corner metrics
            arc_at_nearest = arc_len[nearest_idx]
            arc_corners = arc_len[corner_idx]
            arc_n = arc_at_nearest.unsqueeze(-1)
            arc_c = arc_corners.unsqueeze(0)

            fwd_dist_forward = torch.where(
                arc_c >= arc_n, arc_c - arc_n, track_len - arc_n + arc_c
            )
            fwd_dir_mask = fwd_dir.unsqueeze(-1)
            fwd_dist = torch.where(
                fwd_dir_mask > 0.5, fwd_dist_forward, track_len - fwd_dist_forward
            )
            best_dist, best_k = fwd_dist.min(dim=1)

            no_corner = best_dist >= track_len
            dist_corner = torch.where(
                no_corner, torch.full_like(best_dist, track_len), best_dist
            )
            corner_curv = torch.where(
                no_corner, torch.zeros_like(best_dist), curvature[corner_idx][best_k]
            )

            dist_corner_norm = (dist_corner / cfg.corner_lookahead_max_m).clamp(
                0.0, 1.0
            )
            corner_curv_norm = (corner_curv / cfg.max_curvature).clamp(0.0, 1.0)

            # ── NEW: Spatial Lookahead Trajectory Horizon (8 to 22) ──
            # Sample point positions at fixed forward index offsets (since point density is uniform ~0.05m)
            # offsets correspond roughly to 0.5m, 1.0m, 2.0m, 3.5m, and 5.0m lookahead distance vectors.
            cos_yaw = torch.cos(car_yaw)
            sin_yaw = torch.sin(car_yaw)
            lookahead_offsets = [10, 20, 40, 70, 100]
            lookahead_features = []

            for offset in lookahead_offsets:
                lh_idx = (nearest_idx + idx_sign * offset) % M
                lh_pos = cl[lh_idx]  # [N, 2] world position of lookahead point

                # Compute distance vector from vehicle to lookahead point
                dx = lh_pos[:, 0] - pos_xy[:, 0]
                dy = lh_pos[:, 1] - pos_xy[:, 1]

                # Transform world coordinates to vehicle-local body frame (X=Forward, Y=Lateral Left)
                local_x = dx * cos_yaw + dy * sin_yaw
                local_y = -dx * sin_yaw + dy * cos_yaw

                # Normalize to standard [-1, 1] range using lookahead scalar capacity bounds
                local_x_norm = (local_x / cfg.corner_lookahead_max_m).clamp(-1.0, 1.0)
                local_y_norm = (local_y / cfg.corner_lookahead_max_m).clamp(-1.0, 1.0)

                # Extract and normalize the local track curvature profile at this lookup point
                lh_curv = curvature[lh_idx]
                lh_curv_norm = (lh_curv / cfg.max_curvature).clamp(0.0, 1.0)

                lookahead_features.extend([local_x_norm, local_y_norm, lh_curv_norm])

        obs_list = [
            wheel_vel_norm,
            car_vel_fwd_norm,
            lateral_vel_norm,
            yaw_rate_norm,
            dist_cl_norm,
            rel_heading_norm,
            dist_corner_norm,
            corner_curv_norm,
        ] + lookahead_features

        obs = torch.stack(obs_list, dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0]
        dt = self.cfg.decimation * self.cfg.sim.dt

        r_distance = (
            self.cfg.distance_weight * lin_vel_b / self.cfg.max_velocity_m_s * dt
        )

        forward_speed = lin_vel_b.clamp(min=0.0)

        steer_norm = torch.abs(self._filtered_steer).clamp(0.0, 1.0)
        steer_excess = (steer_norm - self.cfg.steer_deadzone).clamp(min=0.0)
        steer_excess_norm = steer_excess / max(1.0 - self.cfg.steer_deadzone, 1e-6)
        r_steer_shape = -self.cfg.steer_deadzone_weight * steer_excess_norm

        steer_rate = (self._steer_cmd_curr - self._steer_cmd_prev).abs()
        r_steer_rate = -self.cfg.steer_rate_weight * steer_rate

        throttle_rate = (self._throttle_cmd_curr - self._throttle_cmd_prev).abs()
        r_throttle_rate = -self.cfg.throttle_rate_weight * throttle_rate

        wheel_omega_mean = (
            self.robot.data.joint_vel[:, self._all_throttle_ids].abs().mean(dim=-1)
        )
        wheel_vel = wheel_omega_mean * WHEEL_RADIUS
        slip_ratio = (wheel_vel - forward_speed).abs() / forward_speed.clamp(min=1e-3)
        r_slip = -self.cfg.slip_weight * slip_ratio.clamp(0.0, 1.0)

        q = self.robot.data.root_quat_w
        sin_roll = 2.0 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3])
        cos_roll = 1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])
        roll_deg = torch.rad2deg(torch.atan2(sin_roll, cos_roll))
        roll_frac = roll_deg.abs() / 15.0
        r_roll = -self.cfg.roll_weight * roll_frac

        r_alive = torch.full(
            (self.num_envs,), self.cfg.alive_weight, device=self.device
        )

        total = (
            r_alive
            + r_distance
            + r_steer_shape
            + r_steer_rate
            + r_throttle_rate
            + r_slip
            + r_roll
        )
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        wall_hit = self._compute_wall_hit()
        in_grace = self.episode_length_buf < self.cfg.spawn_grace_steps
        self._wall_ever_touched |= wall_hit & ~in_grace
        wall_terminated = self._wall_ever_touched

        q = self.robot.data.root_quat_w
        up_z = 1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])
        flipped = (up_z < 0.3) & ~in_grace

        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0]
        stopped = (
            lin_vel_b < (self.cfg.slow_stop_speed_fraction * self.cfg.max_velocity_m_s)
        ) & (self.episode_length_buf > 45)

        terminated = wall_terminated | flipped | stopped
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        ep_lens = self.episode_length_buf[env_ids].float()
        flipped_mask = self._is_flipped_spawn[env_ids]
        if len(env_ids) > 0:
            len_normal = ep_lens[~flipped_mask]
            len_flipped = ep_lens[flipped_mask]
            if len_normal.numel() > 0:
                self._mean_ep_len_normal = (
                    (1.0 - self._direction_ema_alpha) * self._mean_ep_len_normal
                    + self._direction_ema_alpha * len_normal.mean()
                )
            if len_flipped.numel() > 0:
                self._mean_ep_len_flipped = (
                    (1.0 - self._direction_ema_alpha) * self._mean_ep_len_flipped
                    + self._direction_ema_alpha * len_flipped.mean()
                )

        super()._reset_idx(env_ids)

        n = len(env_ids)
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        speed_vec = torch.zeros(n, device=self.device)
        sz = 0.002
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        pts_normal = list(self.cfg.preset_spawn_points)
        pts_flipped = [(x, y, theta + 180.0) for x, y, theta in pts_normal]
        n_normal = len(pts_normal)
        eps = 1e-6
        p_flipped = (
            self._mean_ep_len_normal
            / (self._mean_ep_len_normal + self._mean_ep_len_flipped + eps)
        ).clamp(0.05, 0.95)
        is_flipped = torch.rand(n, device=self.device) < p_flipped
        self._is_flipped_spawn[env_ids_t] = is_flipped
        idx_normal = torch.randint(0, n_normal, (n,), device="cpu")
        idx_flipped = torch.randint(0, len(pts_flipped), (n,), device="cpu")
        idx = torch.where(is_flipped.cpu(), idx_flipped + n_normal, idx_normal).tolist()
        pts = pts_normal + pts_flipped
        xs = torch.tensor(
            [pts[i][0] * self.cfg.spawn_x_scale for i in idx],
            dtype=torch.float32,
            device=self.device,
        )
        ys = torch.tensor(
            [pts[i][1] * self.cfg.spawn_y_scale for i in idx],
            dtype=torch.float32,
            device=self.device,
        )
        base_yaw = torch.tensor(
            [math.radians(pts[i][2]) for i in idx],
            dtype=torch.float32,
            device=self.device,
        )
        yaw = base_yaw.clone()
        if self.cfg.spawn_yaw_jitter_rad > 0.0:
            yaw = (
                yaw
                + (torch.rand(n, device=self.device) - 0.5)
                * 2.0
                * self.cfg.spawn_yaw_jitter_rad
            )
        spawn_xy = torch.stack([xs, ys], dim=-1)

        default_root_state[:, 0] = spawn_xy[:, 0]
        default_root_state[:, 1] = spawn_xy[:, 1]
        default_root_state[:, 2] = sz

        zeros = torch.zeros(n, device=self.device)
        dq = quat_from_euler_xyz(zeros, zeros, yaw)
        default_root_state[:, 3:7] = dq

        default_root_state[:, 7:] = 0.0
        default_root_state[:, 7] = speed_vec * torch.cos(yaw)
        default_root_state[:, 8] = speed_vec * torch.sin(yaw)

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self._filtered_steer[env_ids] = 0.0
        self._steer_cmd_curr[env_ids] = 0.0
        self._steer_cmd_prev[env_ids] = 0.0
        self._throttle_cmd_curr[env_ids] = 0.0
        self._throttle_cmd_prev[env_ids] = 0.0

        self._wall_ever_touched[env_ids] = False
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
