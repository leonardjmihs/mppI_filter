#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, TransformStamped
from std_msgs.msg import Header
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_pose_stamped
import numpy as np

# run only for vanilla mppi
class SafezoneGoalService(Node):
    def __init__(self):
        super().__init__("safezone_goal_service")

        # Topics (override with params if desired)
        self.declare_parameter("safe_zones_topic", "safe_zones")
        self.declare_parameter("odom_topic", "/dlio/odom_node/odom")
        self.declare_parameter("goal_topic", "goal_pose")
        # Behavior toggles
        self.declare_parameter("fallback_to_robot_frame", True)

        self.safe_zones_topic = self.get_parameter("safe_zones_topic").value
        self.odom_topic       = self.get_parameter("odom_topic").value
        self.goal_topic       = self.get_parameter("goal_topic").value
        self.fallback_to_robot_frame = bool(self.get_parameter("fallback_to_robot_frame").value)

        # Latest data cache
        self._safe_centers = np.empty((0, 2), dtype=float)  # (N,2) in _safe_zones_frame
        self._safe_zones_frame = None                       # frame_id of the safe-zone markers
        self._last_odom: Odometry | None = None

        # TF
        self._tf_buf = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)

        # Pub/Sub
        cb_sub = MutuallyExclusiveCallbackGroup()
        cb_srv = MutuallyExclusiveCallbackGroup()

        self.create_subscription(MarkerArray, self.safe_zones_topic, self._safezones_cb, 10, callback_group=cb_sub)
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, 50, callback_group=cb_sub)
        self.goal_pub = self.create_publisher(PoseStamped, self.goal_topic, 10)

        # Service: /Noooooo
        self.trig_srv = self.create_service(Trigger, "Noooooo", self._on_trigger, callback_group=cb_srv)

        self.get_logger().info("[Noooooo] service ready (nearest safe-zone → /goal_pose).")

    # --- subscribers ---
    def _safezones_cb(self, msg: MarkerArray):
        centres = []
        frame = None
        if msg.markers:
            frame = msg.markers[0].header.frame_id or ""
        for m in msg.markers:
            # SPHERE: center in pose.position
            if m.type == Marker.SPHERE:
                centres.append([m.pose.position.x, m.pose.position.y])
            # SPHERE_LIST: may use points[] or pose as the center (we’ll take both if present)
            elif m.type == Marker.SPHERE_LIST:
                if m.points:
                    for pt in m.points:
                        centres.append([pt.x, pt.y])
                else:
                    centres.append([m.pose.position.x, m.pose.position.y])
            # Any other marker with points[] (e.g., POINTS)
            elif m.points:
                for pt in m.points:
                    centres.append([pt.x, pt.y])
            else:
                # Fallback to pose center if nothing else
                centres.append([m.pose.position.x, m.pose.position.y])

        self._safe_zones_frame = frame
        self._safe_centers = np.asarray(centres, dtype=float) if centres else np.empty((0, 2), dtype=float)

    def _odom_cb(self, msg: Odometry):
        self._last_odom = msg

    # --- service callback ---
    def _on_trigger(self, request: Trigger.Request, response: Trigger.Response):
        if self._last_odom is None:
            response.success = False
            response.message = "No odometry received yet."
            return response
        if self._safe_centers.size == 0:
            response.success = False
            response.message = "No safe zones received yet."
            return response

        # Robot pose in its odom frame
        odom_frame = self._last_odom.header.frame_id or ""
        robot_ps = PoseStamped(
            header=Header(frame_id=odom_frame, stamp=self._last_odom.header.stamp),
            pose=self._last_odom.pose.pose
        )

        # Default: compute & publish in safe-zones frame
        target_frame = self._safe_zones_frame or odom_frame

        # Try to transform robot pose into the safe-zones frame
        use_sz_frame = True
        try:
            if target_frame != odom_frame:
                t: TransformStamped = self._tf_buf.lookup_transform(
                    target_frame, odom_frame, Time(), timeout=Duration(seconds=0.5)
                )
                robot_ps = do_transform_pose_stamped(robot_ps, t)
        except Exception as e:
            if not self.fallback_to_robot_frame:
                response.success = False
                response.message = f"TF lookup failed {odom_frame}->{target_frame}: {e}"
                return response
            # Fall back: compute in robot frame and publish goal in robot frame
            self.get_logger().warn(
                f"[Noooooo] TF unavailable {odom_frame}->{target_frame}; "
                f"falling back to robot frame '{odom_frame}'."
            )
            target_frame = odom_frame
            use_sz_frame = False  # (we’ll compute distances in robot frame)

        # Compute nearest center
        rx, ry = robot_ps.pose.position.x, robot_ps.pose.position.y
        centres = self._safe_centers
        if not use_sz_frame and (self._safe_zones_frame is not None) and (self._safe_zones_frame != odom_frame):
            # Need safe-zone centres in robot frame for distance computation
            try:
                t_sz_to_odom: TransformStamped = self._tf_buf.lookup_transform(
                    odom_frame, self._safe_zones_frame, Time(), timeout=Duration(seconds=0.5)
                )
                # Transform each center as a PoseStamped with identity orientation
                centres_tf = []
                for cx, cy in centres:
                    ps = PoseStamped(header=Header(frame_id=self._safe_zones_frame),
                                     pose=Pose(position=Point(x=float(cx), y=float(cy), z=0.0),
                                               orientation=Quaternion(w=1.0)))
                    ps_o = do_transform_pose_stamped(ps, t_sz_to_odom)
                    centres_tf.append([ps_o.pose.position.x, ps_o.pose.position.y])
                centres = np.asarray(centres_tf, dtype=float)
            except Exception as e:
                response.success = False
                response.message = f"TF fallback failed {self._safe_zones_frame}->{odom_frame}: {e}"
                return response

        dists = np.linalg.norm(centres - np.array([rx, ry])[None, :], axis=1)
        idx = int(np.argmin(dists))
        gx, gy = centres[idx]

        # Publish /goal_pose in target_frame
        goal_msg = PoseStamped(
            header=Header(frame_id=target_frame),
            pose=Pose(
                position=Point(x=float(gx), y=float(gy), z=0.0),
                orientation=Quaternion(w=1.0),
            ),
        )
        self.goal_pub.publish(goal_msg)
        response.success = True
        response.message = f"Published nearest safe-zone as goal: ({gx:.3f}, {gy:.3f}) in frame '{target_frame}'."
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SafezoneGoalService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()