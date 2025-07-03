#!/usr/bin/env python
# src/scripts/roam_with_joy_multi.py

import argparse
import time
from src.roamers.multi_robot_habitat_roamer import MultiRobotHabitatRoamer


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-file-path", default="launch/teleop_multi.launch", type=str)
    parser.add_argument(
        "--hab-env-node-path", default="src/nodes/habitat_env_node.py", type=str
    )
    parser.add_argument(
        "--hab-env-config-path", default="configs/setting_5_configs/pointnav_rgbd-mp3d_with_physics.yaml", type=str
    )
    parser.add_argument("--num-robots", type=int, default=2)
    parser.add_argument("--episode-id", type=str, default="-1")
    parser.add_argument(
        "--scene-id",
        type=str,
        default="data/scene_datasets/mp3d/2t7WUuJeko7/2t7WUuJeko7.glb",
    )
    parser.add_argument(
        "--video-frame-period",
        type=int,
        default=60,
    )
    args = parser.parse_args()

    # start the multi-robot roamer nodes
    roamer = MultiRobotHabitatRoamer(
        launch_file_path=args.launch_file_path,
        hab_env_node_path=args.hab_env_node_path,
        hab_env_config_path=args.hab_env_config_path,
        num_robots=args.num_robots,
        video_frame_period=args.video_frame_period,
    )

    # get to the specified episode
    roamer.roam_until_shutdown(args.episode_id, args.scene_id)


if __name__ == "__main__":
    main()