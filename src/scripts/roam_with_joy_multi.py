# src/scripts/roam_with_joy_multi.py
import argparse
import time
from multiprocessing import Process
from src.roamers.multi_robot_joy_habitat_roamer import MultiRobotJoyHabitatRoamer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-file-path", default="launch/teleop_multi.launch", type=str)
    parser.add_argument("--hab-env-node-path", default="src/nodes/habitat_env_node.py", type=str)
    parser.add_argument("--hab-env-config-path", default="configs/setting_5_configs/pointnav_rgbd-mp3d_with_physics_multi.yaml", type=str)
    parser.add_argument("--num-robots", type=int, default=2)
    parser.add_argument("--episode-id", type=str, default="-1")
    parser.add_argument("--scene-id", type=str, default="data/scene_datasets/mp3d/2t7WUuJeko7/2t7WUuJeko7.glb")
    parser.add_argument("--video-frame-period", type=int, default=60)
    parser.add_argument("--enable-mapping", action="store_true", help="Enable RTAB-Map SLAM")  # 새로 추가
    args = parser.parse_args()

    # Start multi-robot roamer
    roamer = MultiRobotJoyHabitatRoamer(
        launch_file_path=args.launch_file_path,
        hab_env_node_path=args.hab_env_node_path,
        hab_env_config_path=args.hab_env_config_path,
        num_robots=args.num_robots,
        video_frame_period=args.video_frame_period,
        enable_mapping=args.enable_mapping,  # 새로 추가
    )

    # Start roaming
    roamer.roam_until_shutdown(args.episode_id, args.scene_id)

if __name__ == "__main__":
    main()