#!/usr/bin/env python
# src/scripts/simple_map_merger.py

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid

class SimpleMapMerger:
    def __init__(self):
        rospy.init_node('simple_map_merger')
        
        self.map1 = None
        self.map2 = None
        
        # 구독자
        self.sub1 = rospy.Subscriber('/robot_0_hector/map', OccupancyGrid, self.map1_callback)
        self.sub2 = rospy.Subscriber('/robot_1_hector/map', OccupancyGrid, self.map2_callback)
        
        # 발행자
        self.pub = rospy.Publisher('/merged_hector_map', OccupancyGrid, queue_size=1)
        
        # 타이머
        self.timer = rospy.Timer(rospy.Duration(1.0), self.merge_maps)
        
    def map1_callback(self, msg):
        self.map1 = msg
        
    def map2_callback(self, msg):
        self.map2 = msg
        
    def merge_maps(self, event):
        if self.map1 is None or self.map2 is None:
            return
            
        # 간단한 합치기: map1을 기준으로 map2의 점유된 셀들을 추가
        merged = OccupancyGrid()
        merged.header = self.map1.header
        merged.info = self.map1.info
        
        data1 = np.array(self.map1.data)
        data2 = np.array(self.map2.data)
        
        # 두 맵의 크기가 같다고 가정하고 합치기
        if len(data1) == len(data2):
            merged_data = np.maximum(data1, data2)
            merged.data = merged_data.tolist()
            self.pub.publish(merged)

if __name__ == '__main__':
    try:
        SimpleMapMerger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass