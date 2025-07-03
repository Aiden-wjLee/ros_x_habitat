#!/usr/bin/env python3
"""
멀티 로봇 Habitat 인터페이스 - 기본 버전
"""

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

class MultiRobotHabitatInterface:
    def __init__(self):
        rospy.init_node('multi_robot_habitat_interface')
        
        config_path = rospy.get_param('~config_path', '')
        rospy.loginfo(f"Habitat interface initialized with config: {config_path}")
        
        # 기본 퍼블리셔들
        self.rgb_pubs = {}
        self.depth_pubs = {}
        self.cmd_subs = {}
        
        self.setup_interface()
        
    def setup_interface(self):
        """인터페이스 설정"""
        for i in range(2):  # 2개 로봇
            # RGB 퍼블리셔
            self.rgb_pubs[i] = rospy.Publisher(
                f'/robot_{i}/camera/rgb/image_raw', 
                Image, 
                queue_size=1
            )
            
            # Depth 퍼블리셔  
            self.depth_pubs[i] = rospy.Publisher(
                f'/robot_{i}/camera/depth/image_raw', 
                Image, 
                queue_size=1
            )
            
            # 명령 구독자
            self.cmd_subs[i] = rospy.Subscriber(
                f'/robot_{i}/cmd_vel', 
                Twist, 
                lambda msg, robot_id=i: self.cmd_callback(msg, robot_id)
            )
    
    def cmd_callback(self, msg, robot_id):
        """명령 콜백"""
        rospy.loginfo(f"Robot {robot_id} cmd: linear={msg.linear.x:.2f}, angular={msg.angular.z:.2f}")
    
    def run(self):
        """메인 루프"""
        rate = rospy.Rate(30)
        
        rospy.loginfo("Multi-robot Habitat interface running...")
        
        while not rospy.is_shutdown():
            # TODO: Habitat 시뮬레이션 스텝 및 센서 데이터 퍼블리시
            rate.sleep()

def main():
    """메인 함수"""
    try:
        interface = MultiRobotHabitatInterface()
        interface.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Interface error: {e}")

if __name__ == "__main__":
    main()