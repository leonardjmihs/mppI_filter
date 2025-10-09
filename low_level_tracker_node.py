import rclpy, numpy as np
from rclpy.node                import Node
from std_msgs.msg              import Float32MultiArray, ColorRGBA, Bool
from nav_msgs.msg              import Odometry
from geometry_msgs.msg         import Twist, Point, Vector3
from visualization_msgs.msg    import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped

def make_line_marker(pts, frame_id: str, mid: int,
                     rgb=(0.0, 1.0, 1.0), width=0.05, alpha=0.40) -> Marker:
    m                 = Marker()
    m.header.frame_id = frame_id
    m.header.stamp    = rclpy.clock.Clock().now().to_msg()
    m.ns, m.id        = "lqr_preview", mid
    m.type, m.action  = Marker.LINE_STRIP, Marker.ADD
    m.scale           = Vector3(x=width, y=0.0, z=0.0)
    r, g, b           = rgb
    m.color           = ColorRGBA(r=r, g=g, b=b, a=alpha)
    m.points          = [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in pts]
    return m

def unicycle_step(xyθ, u, dt):
    ω, v    = u
    x, y, θ = xyθ
    x += v * np.cos(θ) * dt
    y += v * np.sin(θ) * dt
    θ += ω * dt
    return np.array([x, y, θ])

def yaw_from_quat(q):
    return float(np.arctan2(2*(q.w*q.z + q.x*q.y),
                            1 - 2*(q.y*q.y + q.z*q.z)))


# ─── main node ───────────────────────────────────────────────────────────
class Tracker(Node):
    def __init__(self):
        super().__init__("tracker_node")

        # ▸ basic params ---------------------------------------------------
        self.dt   = self.declare_parameter("dt", 0.01).value
        self.Q  = np.diag([ 3.0,  3.0,  0.2])  
        self.R  = np.diag([ 100.0, 1.0])       
        self.Qf = np.diag([18.0, 18.0,  2.0])   
        self.vmax = self.declare_parameter("v_max",  1.0).value
        self.vmin = self.declare_parameter("v_min", -1.0).value
        self.wmax = self.declare_parameter("w_max",  0.75).value
        self.wmin = self.declare_parameter("w_min", -0.75).value

        # ▸ state ----------------------------------------------------------
        self.u_ref = np.empty((0, 2), dtype=np.float32)   # [ω, v] plan
        self.x_ref = np.empty((0, 3), dtype=np.float32)   # rollout states
        self.K_seq = []                                   # list of 2×3 gains
        self.idx   = 0
        self.odom  = None
        self.blend_horizon = self.declare_parameter("blend_horizon", 15).value  # how many samples to blend when starting a new plan
        self.u_prev = np.array([0.0, 0.0], dtype=np.float32)                 


        self.create_subscription(Float32MultiArray, "mppi_plan", self.plan_cb, 10)
        self.create_subscription(Odometry, "/dlio/odom_node/odom", self.odom_cb, 60)
        self.goal_xy_tol = self.declare_parameter("goal_xy_tol", 0.5).value
        self.goal_xy = None                       # will hold np.array([x, y])
        self.create_subscription(PoseStamped, "goal_pose", self.goal_cb, 10)
        self.create_subscription(PoseStamped, "goal", self.goal_cb, 10)

        self.safe_zones = [] 
        self.robot_radius = self.declare_parameter("robot_radius", 0.30).value
        self.create_subscription(
            MarkerArray, "safe_zones", self.safe_zones_cb, 10
        )
        self.contingency_active = False
        self.create_subscription(
            Bool, "contingency_active", self.cont_cb, 10
        )
        self.cmd_pub  = self.create_publisher(Twist,       "cmd_vel",        10)
        self.path_pub = self.create_publisher(MarkerArray, "track_mpc_path", 10)

        self.create_timer(self.dt, self.ctrl_step)
        self.get_logger().info("Tracker up – waiting for plan…")

    def cont_cb(self, msg: Bool):
        self.contingency_active = msg.data

    def goal_cb(self, msg: PoseStamped):
        self.goal_xy = np.array([msg.pose.position.x, msg.pose.position.y], dtype=float)
        self.get_logger().info(f"New goal at {self.goal_xy}", throttle_duration_sec=1.0)

    def safe_zones_cb(self, msg: MarkerArray):
        zs = []
        for m in msg.markers:
            x = m.pose.position.x
            y = m.pose.position.y
            r = m.scale.x / 2.0
            zs.append((x, y, r))
        self.safe_zones = zs

    def plan_cb(self, msg: Float32MultiArray):
        raw = np.asarray(msg.data, dtype=np.float32)
        if raw.size == 0 or raw.size % 2:
            self.get_logger().warn("✘ malformed plan – ignored");  return

        # self.plan_time = self.get_clock().now()

        coarse = raw.reshape(-1, 2)
        steps  = 20               
        Nc     = coarse.shape[0]

        if Nc > 1:
            idx_coarse = np.arange(Nc)
            idx_fine   = np.linspace(0, Nc-1, Nc*steps)


            u_ref_fine = np.vstack([
                np.interp(idx_fine, idx_coarse, coarse[:, d])
                for d in range(coarse.shape[1])
            ]).T
        else:
            u_ref_fine = np.repeat(coarse, steps, axis=0)

        # self.u_ref = u_ref_fine.astype(np.float32)
        # --- Smoothly blend the first few controls with the last published command
        u_new = u_ref_fine.astype(np.float32)
        if hasattr(self, "u_prev") and self.blend_horizon > 0:
            bh = min(self.blend_horizon, len(u_new))
            alpha = np.linspace(1.0, 0.0, bh, endpoint=False).reshape(-1, 1)
            beta = 1.0 - alpha
            u_new[:bh] = alpha * self.u_prev + beta * u_new[:bh]

        self.u_ref = u_new
        N = len(self.u_ref)

        now = self.get_clock().now()

        if hasattr(self, "plan_time"):
            elapsed_old   = (now - self.plan_time).nanoseconds * 1e-9
            steps_elapsed = int(elapsed_old / self.dt)

            if steps_elapsed > 0:
                u_new = u_new[steps_elapsed:]
            if len(u_new) == 0:                     
                u_new = np.vstack([self.u_prev])   

            self.u_ref = u_new
            N          = len(self.u_ref)

            self.idx       = 0
            self.plan_time = now
        else:
            self.u_ref     = u_new
            N              = len(self.u_ref)
            self.idx       = 0
            self.plan_time = now


        if self.odom is None:
            self.get_logger().warn("Plan arrived before odom; buffering...")
            return

        pose = self.odom.pose.pose
        x0   = np.array([pose.position.x,
                        pose.position.y,
                        yaw_from_quat(pose.orientation)], dtype=np.float32)

        self.x_ref = np.empty((N+1, 3), dtype=np.float32)
        self.x_ref[0] = x0
        for k in range(N):
            self.x_ref[k+1] = unicycle_step(self.x_ref[k], self.u_ref[k], self.dt)

        A = np.zeros((N, 3, 3), dtype=np.float32)
        B = np.zeros((N, 3, 2), dtype=np.float32)
        for k in range(N):
            θ = self.x_ref[k, 2]
            v = self.u_ref[k, 1]
            A[k] = np.array([[1, 0, -v*self.dt*np.sin(θ)],
                            [0, 1,  v*self.dt*np.cos(θ)],
                            [0, 0,  1]], dtype=np.float32)
            B[k] = np.array([[0,              self.dt*np.cos(θ)],
                            [0,              self.dt*np.sin(θ)],
                            [self.dt,        0               ]], dtype=np.float32)

        P = self.Qf.copy()
        self.K_seq = [None] * N
        for k in reversed(range(N)):
            S = self.R + B[k].T @ P @ B[k]
            K = np.linalg.solve(S, B[k].T @ P @ A[k])  # 2×3
            self.K_seq[k] = K
            P = self.Q + A[k].T @ P @ (A[k] - B[k] @ K)

        # self.get_logger().info(f"✔ new TV-LQR sequence built (N={N})")

    # ─── odometry update ─────────────────────────────────────────────
    def odom_cb(self, msg: Odometry):
        self.odom = msg

    # ─── controller timer ────────────────────────────────────────────
    def ctrl_step(self):
        if self.odom is None:           
            self.cmd_pub.publish(Twist())
            return

        
        if self.contingency_active:
            px = self.odom.pose.pose.position.x
            py = self.odom.pose.pose.position.y
            for (cx, cy, r_zone) in self.safe_zones:
                if r_zone <= self.robot_radius:
                    continue
                if np.hypot(px - cx, py - cy) <= (r_zone - self.robot_radius):
                    self.get_logger().info("✔ inside safe zone – stopping")
                    self.cmd_pub.publish(Twist())
                    self.idx = len(self.u_ref)        
                    # self.contingency_active = False    
                    return
                
        now       = self.get_clock().now()
        
        if not hasattr(self, 'plan_time'):
            self.cmd_pub.publish(Twist()); return
        
        elapsed   = (now - self.plan_time).nanoseconds * 1e-9
        idx_float = elapsed / self.dt
        self.idx  = min(int(idx_float), len(self.u_ref)-1)

        if self.idx >= len(self.u_ref):
            self.cmd_pub.publish(Twist())
            return

        pose = self.odom.pose.pose
        x_now = np.array([pose.position.x,
                          pose.position.y,
                          yaw_from_quat(pose.orientation)])

        if self.goal_xy is not None:
            if np.linalg.norm(x_now[:2] - self.goal_xy) < self.goal_xy_tol:
                self.idx = len(self.u_ref)     
                self.cmd_pub.publish(Twist())  
                return

        u_r  = self.u_ref[self.idx]
        x_r  = self.x_ref[self.idx]
        K    = self.K_seq[self.idx]
        err         = x_now - x_r
        err[2]      = (err[2] + np.pi) % (2*np.pi) - np.pi 
        u           = u_r - K @ err

        ω = float(np.clip(u[0], self.wmin, self.wmax))
        v = float(np.clip(u[1], self.vmin, self.vmax))

        tw          = Twist()
        tw.linear.x = v
        tw.angular.z= ω
        self.cmd_pub.publish(tw)

        self.u_prev = np.array([tw.angular.z, tw.linear.x], dtype=np.float32)

        # RViz preview 
        sim = x_now.copy()
        pts = [sim[:2]]
        look = range(self.idx, min(self.idx+100, len(self.u_ref)))
        for j in look:
            sim = unicycle_step(sim, self.u_ref[j], self.dt)
            pts.append(sim[:2])
        ma = MarkerArray();  ma.markers = [make_line_marker(pts, "odom_dlio", 0)]
        self.path_pub.publish(ma)

        # self.idx += 1


# ─── entry point ───────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(Tracker())
