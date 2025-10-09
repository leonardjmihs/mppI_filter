#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from numpy.linalg import norm
from math import radians, atan2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker
from scipy.ndimage import distance_transform_edt


class SafeZoneNode(Node):
    def __init__(self) -> None:
        super().__init__("safe_zone_node")
        self.declare_parameter("fov_angle_deg", 360.0)
        self.declare_parameter("fov_range_m", 100.0)
        self.declare_parameter("num_rays", 1000)
        self.declare_parameter("clearance_m", 0.0)

        self.fov_angle_deg = self.get_parameter("fov_angle_deg").value
        self.fov_range = self.get_parameter("fov_range_m").value
        self.num_rays = self.get_parameter("num_rays").value
        self.clearance_m = self.get_parameter("clearance_m").value

        self.adversary_pose = [0.0, 0.0, 0.0]
        half_fov = radians(self.fov_angle_deg) / 2.0
        angles = np.linspace(-half_fov, half_fov, self.num_rays, dtype=np.float32)
        self.ray_unit = np.stack((np.cos(angles), np.sin(angles)), axis=1)

        self._clearance_cells = 0
        self._step_cache = {}

        self.create_subscription(OccupancyGrid, "/costmap/costmap", self.costmap_cb, 10)
        self.create_subscription(PoseStamped, "/adversary_pose", self.adversary_cb, 10)

        self.mask_pub = self.create_publisher(OccupancyGrid, "/occlusion_mask", 10)
        self.fov_marker_pub = self.create_publisher(Marker, "/fov_marker", 1)
        self.adv_marker_pub = self.create_publisher(Marker, "/adversary_marker", 1)

    def adversary_cb(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        q = msg.pose.orientation
        yaw = atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        self.adversary_pose = [p.x, p.y, yaw]

    def _step_table_for_yaw(self, yaw: float, max_steps: int) -> np.ndarray:
        key = (round(yaw, 3), max_steps)
        if key in self._step_cache:
            return self._step_cache[key]
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        dirs = self.ray_unit @ rot.T
        dirs /= norm(dirs, axis=1, keepdims=True)
        steps = np.arange(1, max_steps + 1, dtype=np.float32)[:, None, None]
        tbl = np.round(steps * dirs[None, :, :]).astype(np.int16)
        self._step_cache[key] = tbl
        return tbl

    def costmap_cb(self, msg: OccupancyGrid) -> None:
        h, w = msg.info.height, msg.info.width
        res = msg.info.resolution
        org_x, org_y = msg.info.origin.position.x, msg.info.origin.position.y
        cost = np.frombuffer(msg.data, dtype=np.int8).reshape((h, w))

        max_steps = int(self.fov_range / res + 0.5)
        self._clearance_cells = int(np.ceil(self.clearance_m / res))

        adv_x, adv_y, adv_yaw = self.adversary_pose
        ix = int((adv_x - org_x) / res)
        iy = int((adv_y - org_y) / res)
        step_xy = self._step_table_for_yaw(adv_yaw, max_steps)

        xs = ix + step_xy[..., 0]
        ys = iy + step_xy[..., 1]
        inside = (0 <= xs) & (xs < w) & (0 <= ys) & (ys < h)

        samples = np.zeros(xs.shape, dtype=np.int16)
        samples[inside] = cost[ys[inside], xs[inside]]
        samples[~inside] = 101

        hit = samples >= 99
        first_hit = hit.argmax(axis=0)
        first_hit[~hit.any(axis=0)] = samples.shape[0]

        ray_range = np.arange(samples.shape[0])[:, None]
        visible = (ray_range < first_hit[None, :]) & inside

        lethal = cost >= 99
        traversable = cost == 0

        seen = np.zeros_like(cost, dtype=bool)
        lin_idx = (ys * w + xs).ravel()
        seen.flat[lin_idx[visible.ravel()]] = True

        k = self._clearance_cells
        clear = distance_transform_edt(~lethal) > k if k > 0 else np.ones_like(cost, dtype=bool)

        safe = traversable & (~seen) & clear
        mask = np.zeros_like(cost, dtype=np.uint8)
        mask[safe] = 50
        mask[lethal] = 100

        out = OccupancyGrid()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = msg.header.frame_id
        out.info = msg.info
        out.data = mask.astype(np.int8).ravel().tolist()
        self.mask_pub.publish(out)

        self._publish_fov_marker(adv_x, adv_y, adv_yaw, msg.header.frame_id)
        self._publish_adversary_marker(adv_x, adv_y, msg.header.frame_id)

    def _publish_fov_marker(self, x, y, yaw, frame_id):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id = "fov", 0
        m.type = Marker.TRIANGLE_LIST
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color.a, m.color.r, m.color.g, m.color.b = 0.3, 0.0, 1.0, 1.0
        m.pose.orientation.w = 1.0
        num = 36
        half = radians(self.fov_angle_deg) / 2.0
        pts = [Point(x=x, y=y, z=0.0)]
        for ang in np.linspace(yaw - half, yaw + half, num):
            pts.append(Point(x=x + self.fov_range * np.cos(ang), y=y + self.fov_range * np.sin(ang), z=0.0))
        for i in range(1, len(pts) - 1):
            m.points.extend([pts[0], pts[i], pts[i + 1]])
        self.fov_marker_pub.publish(m)

    def _publish_adversary_marker(self, x, y, frame_id):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id = "adversary", 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.0
        m.scale.x = m.scale.y = m.scale.z = 0.4
        m.color.a, m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0, 0.0
        m.pose.orientation.w = 1.0
        self.adv_marker_pub.publish(m)


def main() -> None:
    rclpy.init()
    node = SafeZoneNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
