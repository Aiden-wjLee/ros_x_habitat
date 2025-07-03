#!/usr/bin/env python
# src/roamers/multi_robot_habitat_roamer.py

import roslaunch
import rospy
import shlex
import time
from geometry_msgs.msg import Twist
from subprocess import Popen
from ros_x_habitat.srv import Roam
from src.constants.constants import PACKAGE_NAME, ServiceNames


class MultiRobotHabitatRoamer:
    r"""
    Class to allow free-roaming in a Habitat environment through keyboard
    control for multiple robots.
    """

    def __init__(
        self,
        launch_file_path: str,
        hab_env_node_path: str,
        hab_env_config_path: str,
        num_robots: int,
        video_frame_period: int,
    ):
        r"""
        :param launch_file_path: path to the launch file which launches the
            keyboard controller nodes and other nodes for control and visualization
        :param hab_env_node_path: path to the Habitat env node file
        :param hab_env_config_path: path to the Habitat env config file
        :param num_robots: number of robots to spawn
        :param video_frame_period: period at which to record a frame; measured in
            steps per frame
        """
        # create a node
        rospy.init_node("multi_robot_habitat_roamer", anonymous=True)

        self.num_robots = num_robots
        self.env_processes = []
        self.roam_services = []

        # start the launch file
        # code adapted from http://wiki.ros.org/roslaunch/API%20Usage
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)
        self.launch = roslaunch.parent.ROSLaunchParent(self.uuid, [launch_file_path])
        self.launch.start()
        rospy.loginfo(f"launch file {launch_file_path} started")

        # start the env nodes for each robot
        for robot_id in range(self.num_robots):
            robot_name = f"robot_{robot_id}"
            env_node_args = shlex.split(
                f"python {hab_env_node_path} --node-name {robot_name}_env_node --task-config {hab_env_config_path} --enable-physics-sim --use-continuous-agent --robot-namespace {robot_name}"
            )
            env_process = Popen(env_node_args)
            self.env_processes.append(env_process)

            # set up roam service client for each robot
            roam_service_name = (
                f"{PACKAGE_NAME}/{robot_name}_env_node/{ServiceNames.ROAM}"
            )
            roam_service = rospy.ServiceProxy(roam_service_name, Roam)
            self.roam_services.append(roam_service)

        # register frame period
        self.video_frame_period = video_frame_period

        # wait a bit for all nodes to start up
        time.sleep(5)

    def roam_until_shutdown(self, episode_id_last: str = "-1", scene_id_last: str = ""):
        r"""
        Roam in a specified scene until shutdown signalled.
        :param episode_id_last: last episode's ID before the one to roam in.
        :param scene_id_last: last episode's scene ID before the one to
            roam in.
        """
        # code adapted from http://wiki.ros.org/roslaunch/API%20Usage
        try:
            # Initialize roaming for all robots
            for robot_id, roam_service in enumerate(self.roam_services):
                roam_service_name = f"{PACKAGE_NAME}/robot_{robot_id}_env_node/{ServiceNames.ROAM}"
                rospy.wait_for_service(roam_service_name)
                try:
                    resp = roam_service(
                        episode_id_last, scene_id_last, True, self.video_frame_period
                    )
                    assert resp.ack
                    rospy.loginfo(f"Robot {robot_id} roaming started")
                except rospy.ServiceException:
                    rospy.logerr(f"Failed to initiate the roam service for robot {robot_id}!")

            self.launch.spin()
        finally:
            # After Ctrl+C, stop all nodes from running
            for env_process in self.env_processes:
                env_process.kill()
            self.launch.shutdown()