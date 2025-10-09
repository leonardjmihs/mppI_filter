import rclpy
import numpy as np
import random

from math import cos, sin, atan2, radians
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from scipy.ndimage import label



class ZoneGroup(Node):
    def __init__(self):
        super().__init__('zone_group_node')

        self.occlusion_sub = self.create_subscription(
            OccupancyGrid,
            '/occlusion_mask',
            self.occlusion_callback,
            10
        )

        self.groups_pub = self.create_publisher(
            Marker,
            '/group_points',
            10
        )

        self.groups_mask_pub = self.create_publisher(
            MarkerArray,
            '/group_masks',
            10
        )

        self.get_logger().info('Started the Safe Zones grouping node')

    def occlusion_callback(self, msg: OccupancyGrid):
        h = msg.info.height
        w = msg.info.width
        res = msg.info.resolution
        org = msg.info.origin

        # print(f'height = {h}, width = {w}, res = {res}')

        cost = np.array(msg.data, dtype=np.int8).reshape((h, w))

        group_dict, center_indices = self.extract_groups(cost)
        center_points = [self.grid_to_world(i, j, org.position.x, org.position.y, res) for i,j in center_indices]

        # self._publish_center_markers(center_points, msg.header.frame_id)
        self._publish_seg_markers(group_dict, org, res, msg.header.frame_id)


    # Function to group the safezones
    def extract_groups(self, cost):
        mask = (cost == 50)
        structure = np.ones((3,3), dtype=int)
        labeled_array, num_groups = label(mask, structure=structure)
        
        group_dict = {}
        center_indices = []

        for group_id in range(1, num_groups+1):
            indices = np.argwhere(labeled_array == group_id)

            if len(indices) >= 25:
                group_dict[group_id] = [tuple(idx) for idx in indices]

                cen_i = float(np.floor(np.mean(indices[:, 0])))
                cen_j = float(np.floor(np.mean(indices[:, 1])))
                center_indices.append((cen_i, cen_j))

        return group_dict, center_indices
    
    # Function to convert indices to world points
    def grid_to_world(self, i, j, org_x, org_y, res):
        xw = org_x + j*res + res/2.0
        yw = org_y + i*res + res/2.0
        return (xw, yw)
    
    # Function to publish safezones with color segmentations
    def _publish_seg_markers(self, group_dict, org, res, frame_id):
        marker_array = MarkerArray()

        for group_id, indices in group_dict.items():
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'group_mask'
            marker.id = group_id
            marker.type = Marker.CUBE_LIST
            marker.action = Marker.ADD
            marker.scale.x = marker.scale.y = marker.scale.z = res
            marker.pose.orientation.w = 1.0

            random.seed(group_id)
            marker.color.r = random.random()
            marker.color.g = random.random()
            marker.color.b = random.random()
            marker.color.a = 0.7

            for i,j in indices:
                x, y = self.grid_to_world(i, j, org.position.x, org.position.y, res)
                pt = Point(x=x, y=y, z=0.0)
                marker.points.append(pt)

            marker_array.markers.append(marker)

        self.groups_mask_pub.publish(marker_array)

    # Function to publish the center points of the safezones.
    def _publish_center_markers(self, points, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'Center_points'
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.scale.x = marker.scale.y = marker.scale.z = 0.25
        marker.color.r , marker.color.g, marker.color.b, marker.color.a = 0.0, 0.0, 1.0, 1.0

        for x,y in points:

            pt = Point()
            pt.x = x
            pt.y = y
            pt.z = 0.0
            marker.points.append(pt)

        self.groups_pub.publish(marker)
        

def main():
    rclpy.init()
    node = ZoneGroup()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
