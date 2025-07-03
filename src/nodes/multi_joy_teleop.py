#!/usr/bin/env python
# src/nodes/multi_joy_teleop.py

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Int32, String
import time


class MultiJoyTeleop:
    """
    Multi-robot joystick teleop node that allows controlling multiple robots
    with a single joystick by switching between active robots.
    """

    def __init__(self):
        rospy.init_node('multi_joy_teleop', anonymous=True)
        
        # Parameters
        self.num_robots = rospy.get_param('~num_robots', 2)
        self.active_robot = rospy.get_param('~active_robot', 0)
        self.linear_scale = rospy.get_param('~linear_scale', 0.5)
        self.angular_scale = rospy.get_param('~angular_scale', 1.0)
        self.enable_formation = rospy.get_param('~enable_formation', False)
        self.formation_type = rospy.get_param('~formation_type', 'line')
        
        # Button mappings
        self.switch_robot_button = rospy.get_param('~switch_robot_button', 4)
        self.next_robot_button = rospy.get_param('~next_robot_button', 5)
        self.stop_all_button = rospy.get_param('~stop_all_button', 6)
        self.formation_toggle_button = rospy.get_param('~formation_toggle_button', 7)
        
        # Axis mappings
        self.linear_axis = rospy.get_param('~linear_axis', 1)
        self.angular_axis = rospy.get_param('~angular_axis', 0)
        
        # State variables
        self.last_button_state = {}
        self.formation_mode = self.enable_formation
        self.last_twist_time = time.time()
        self.twist_timeout = 0.5  # Stop robot if no command for 0.5s
        
        # Publishers for each robot
        self.robot_publishers = {}
        for robot_id in range(self.num_robots):
            topic = f"robot_{robot_id}/cmd_vel"
            self.robot_publishers[robot_id] = rospy.Publisher(topic, Twist, queue_size=10)
        
        # Publishers for status and control
        self.active_robot_pub = rospy.Publisher('active_robot', Int32, queue_size=10, latch=True)
        self.status_pub = rospy.Publisher('teleop_status', String, queue_size=10)
        self.formation_cmd_pub = rospy.Publisher('formation_cmd_vel', Twist, queue_size=10)
        
        # Subscriber
        self.joy_sub = rospy.Subscriber('joy', Joy, self.joy_callback, queue_size=1)
        
        # Initialize active robot publisher
        self.publish_active_robot()
        self.publish_status("Multi-robot teleop initialized")
        
        # Timer for safety timeout
        self.safety_timer = rospy.Timer(rospy.Duration(0.1), self.safety_callback)
        
        rospy.loginfo(f"Multi-robot teleop started with {self.num_robots} robots")
        rospy.loginfo(f"Active robot: {self.active_robot}")
        rospy.loginfo(f"Formation mode: {'enabled' if self.formation_mode else 'disabled'}")

    def joy_callback(self, joy_msg):
        """Handle joystick input"""
        try:
            # Handle button presses
            self.handle_buttons(joy_msg)
            
            # Create twist message from joystick axes
            twist = Twist()
            if len(joy_msg.axes) > max(self.linear_axis, self.angular_axis):
                twist.linear.x = joy_msg.axes[self.linear_axis] * self.linear_scale
                twist.angular.z = joy_msg.axes[self.angular_axis] * self.angular_scale
            
            # Send commands based on current mode
            if self.formation_mode:
                self.handle_formation_control(twist)
            else:
                self.handle_individual_control(twist)
                
            self.last_twist_time = time.time()
            
        except Exception as e:
            rospy.logerr(f"Error in joy callback: {e}")

    def handle_buttons(self, joy_msg):
        """Handle button presses for robot switching and mode changes"""
        for i, button in enumerate(joy_msg.buttons):
            button_pressed = button and not self.last_button_state.get(i, False)
            
            if button_pressed:
                if i == self.switch_robot_button:
                    self.switch_to_previous_robot()
                elif i == self.next_robot_button:
                    self.switch_to_next_robot()
                elif i == self.stop_all_button:
                    self.stop_all_robots()
                elif i == self.formation_toggle_button:
                    self.toggle_formation_mode()
            
            self.last_button_state[i] = button

    def handle_individual_control(self, twist):
        """Send commands to individual active robot"""
        if 0 <= self.active_robot < self.num_robots:
            self.robot_publishers[self.active_robot].publish(twist)

    def handle_formation_control(self, twist):
        """Send commands for formation control"""
        self.formation_cmd_pub.publish(twist)
        
        # Simple formation implementation
        if self.formation_type == "line":
            for i in range(self.num_robots):
                robot_twist = Twist()
                if i == 0:  # Leader
                    robot_twist = twist
                else:  # Followers
                    robot_twist.linear.x = twist.linear.x * 0.8
                    robot_twist.linear.y = twist.linear.y * 0.8
                    robot_twist.angular.z = twist.angular.z
                
                self.robot_publishers[i].publish(robot_twist)
                
        elif self.formation_type == "triangle":
            spacing = 0.1
            for i in range(min(3, self.num_robots)):
                robot_twist = Twist()
                if i == 0:  # Leader
                    robot_twist = twist
                elif i == 1:  # Left follower
                    robot_twist.linear.x = twist.linear.x * 0.8
                    robot_twist.linear.y = twist.linear.y * 0.8 + spacing
                    robot_twist.angular.z = twist.angular.z
                elif i == 2:  # Right follower
                    robot_twist.linear.x = twist.linear.x * 0.8
                    robot_twist.linear.y = twist.linear.y * 0.8 - spacing
                    robot_twist.angular.z = twist.angular.z
                
                self.robot_publishers[i].publish(robot_twist)

    def switch_to_next_robot(self):
        """Switch to next robot"""
        self.active_robot = (self.active_robot + 1) % self.num_robots
        self.publish_active_robot()
        self.publish_status(f"Switched to robot {self.active_robot}")
        rospy.loginfo(f"Active robot: {self.active_robot}")

    def switch_to_previous_robot(self):
        """Switch to previous robot"""
        self.active_robot = (self.active_robot - 1) % self.num_robots
        self.publish_active_robot()
        self.publish_status(f"Switched to robot {self.active_robot}")
        rospy.loginfo(f"Active robot: {self.active_robot}")

    def toggle_formation_mode(self):
        """Toggle between formation and individual control"""
        self.formation_mode = not self.formation_mode
        mode_str = "formation" if self.formation_mode else "individual"
        self.publish_status(f"Switched to {mode_str} control mode")
        rospy.loginfo(f"Control mode: {mode_str}")

    def stop_all_robots(self):
        """Emergency stop for all robots"""
        stop_twist = Twist()  # All zeros
        for i in range(self.num_robots):
            self.robot_publishers[i].publish(stop_twist)
        self.publish_status("Emergency stop - all robots stopped")
        rospy.logwarn("Emergency stop activated")

    def publish_active_robot(self):
        """Publish current active robot ID"""
        msg = Int32()
        msg.data = self.active_robot
        self.active_robot_pub.publish(msg)

    def publish_status(self, message):
        """Publish status message"""
        msg = String()
        msg.data = message
        self.status_pub.publish(msg)

    def safety_callback(self, event):
        """Safety timeout callback - stop robots if no recent commands"""
        time_since_last_twist = time.time() - self.last_twist_time
        if time_since_last_twist > self.twist_timeout:
            # Send zero velocity to all robots for safety
            stop_twist = Twist()
            for i in range(self.num_robots):
                self.robot_publishers[i].publish(stop_twist)


def main():
    try:
        teleop = MultiJoyTeleop()
        
        rospy.loginfo("Multi-robot joystick teleop ready!")
        rospy.loginfo("Controls:")
        rospy.loginfo("  Left stick: Move active robot")
        rospy.loginfo("  L1/LB: Previous robot")
        rospy.loginfo("  R1/RB: Next robot") 
        rospy.loginfo("  Back/Select: Emergency stop all")
        rospy.loginfo("  Start: Toggle formation mode")
        
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in multi-robot teleop: {e}")


if __name__ == '__main__':
    main()