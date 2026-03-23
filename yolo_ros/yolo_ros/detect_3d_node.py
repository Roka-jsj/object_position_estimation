# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import time
from typing import List, Tuple

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from tf2_ros import TransformException
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from yolo_msgs.msg import BoundingBox3D
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.srv import GetTargetPosition


class Detect3DNode(LifecycleNode):
    """ROS 2 Lifecycle Node for parameter-driven 3D object detection."""

    def __init__(self) -> None:
        super().__init__("bbox3d_node")

        self.declare_parameter("color_info_topic", "color_info")
        self.declare_parameter("depth_image_topic", "depth_image")
        self.declare_parameter("depth_info_topic", "depth_info")
        self.declare_parameter("detections_topic", "detections")
        self.declare_parameter("detections_3d_topic", "detections_3d")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("depth_image_units_divisor", 1000)
        self.declare_parameter("depth_image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("service_name", "get_target_position")
        self.declare_parameter("cache_timeout_sec", 5.0)
        self.declare_parameter("match_substring", True)
        self.declare_parameter("use_tf", True)

        self._color_info = None
        self._target_cache = {}
        self.cv_bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = None
        self.tf_broadcaster = None
        self._color_info_sub = None
        self._srv = None
        self._synchronizer = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.color_info_topic = self.get_parameter("color_info_topic").get_parameter_value().string_value
        self.depth_image_topic = self.get_parameter("depth_image_topic").get_parameter_value().string_value
        self.depth_info_topic = self.get_parameter("depth_info_topic").get_parameter_value().string_value
        self.detections_topic = self.get_parameter("detections_topic").get_parameter_value().string_value
        self.detections_3d_topic = self.get_parameter("detections_3d_topic").get_parameter_value().string_value
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value
        self.depth_image_units_divisor = self.get_parameter("depth_image_units_divisor").get_parameter_value().integer_value
        self.cache_timeout_sec = self.get_parameter("cache_timeout_sec").get_parameter_value().double_value
        self.match_substring = self.get_parameter("match_substring").get_parameter_value().bool_value
        self.use_tf = self.get_parameter("use_tf").get_parameter_value().bool_value

        dimg_reliability = self.get_parameter("depth_image_reliability").get_parameter_value().integer_value
        dinfo_reliability = self.get_parameter("depth_info_reliability").get_parameter_value().integer_value

        self.depth_image_qos_profile = QoSProfile(
            reliability=dimg_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.depth_info_qos_profile = QoSProfile(
            reliability=dinfo_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self._pub = self.create_publisher(DetectionArray, self.detections_3d_topic, 10)

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        srv_name = self.get_parameter("service_name").get_parameter_value().string_value
        self._srv = self.create_service(GetTargetPosition, srv_name, self.on_service)
        self._color_info_sub = self.create_subscription(
            CameraInfo,
            self.color_info_topic,
            self._on_color_info,
            10,
        )

        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            self.depth_image_topic,
            qos_profile=self.depth_image_qos_profile,
        )
        self.depth_info_sub = message_filters.Subscriber(
            self,
            CameraInfo,
            self.depth_info_topic,
            qos_profile=self.depth_info_qos_profile,
        )
        self.detections_sub = message_filters.Subscriber(
            self,
            DetectionArray,
            self.detections_topic,
        )

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.depth_sub, self.depth_info_sub, self.detections_sub),
            10,
            0.5,
        )
        self._synchronizer.registerCallback(self.on_detections)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        if self._color_info_sub is not None:
            self.destroy_subscription(self._color_info_sub)
            self._color_info_sub = None

        self.destroy_subscription(self.depth_sub.sub)
        self.destroy_subscription(self.depth_info_sub.sub)
        self.destroy_subscription(self.detections_sub.sub)
        self.depth_sub = None
        self.depth_info_sub = None
        self.detections_sub = None
        self._synchronizer = None

        if self._srv is not None:
            self.destroy_service(self._srv)
            self._srv = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")
        self.destroy_publisher(self._pub)
        self.tf_listener = None
        self.tf_broadcaster = None
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_shutdown(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def _on_color_info(self, msg: CameraInfo) -> None:
        self._color_info = msg

    def _get_color_to_depth_scale(self, depth_image: np.ndarray) -> Tuple[float, float]:
        if self._color_info is None:
            return 1.0, 1.0

        color_w = int(self._color_info.width)
        color_h = int(self._color_info.height)
        if color_w <= 0 or color_h <= 0:
            return 1.0, 1.0

        depth_h, depth_w = depth_image.shape[:2]
        if color_w == depth_w and color_h == depth_h:
            return 1.0, 1.0

        return depth_w / float(color_w), depth_h / float(color_h)

    def on_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        detections_msg: DetectionArray,
    ) -> None:
        new_detections_msg = DetectionArray()
        new_detections_msg.header = detections_msg.header
        new_detections_msg.detections = self.process_detections(
            depth_msg,
            depth_info_msg,
            detections_msg,
        )
        self._pub.publish(new_detections_msg)
        self._update_cache(new_detections_msg)
        self._publish_object_tf(new_detections_msg)

    def process_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        detections_msg: DetectionArray,
    ) -> List[Detection]:
        if not detections_msg.detections:
            return []

        transform = None
        if self.use_tf:
            transform = self.get_transform(depth_info_msg.header.frame_id)
            if transform is None:
                self.get_logger().warn(
                    f"TF not available. Keeping bbox3d in frame '{depth_info_msg.header.frame_id}'"
                )

        depth_image = self.cv_bridge.imgmsg_to_cv2(
            depth_msg,
            desired_encoding="passthrough",
        )
        new_detections: List[Detection] = []

        for detection in detections_msg.detections:
            bbox3d = self.convert_bb_to_3d(depth_image, depth_info_msg, detection)
            if bbox3d is None:
                continue

            new_detections.append(detection)
            if transform is not None:
                bbox3d = Detect3DNode.transform_3d_box(bbox3d, transform[0], transform[1])
                bbox3d.frame_id = self.target_frame
            else:
                bbox3d.frame_id = depth_info_msg.header.frame_id

            distance = float(
                np.sqrt(
                    bbox3d.center.position.x * bbox3d.center.position.x
                    + bbox3d.center.position.y * bbox3d.center.position.y
                    + bbox3d.center.position.z * bbox3d.center.position.z
                )
            )

            new_detections[-1].bbox3d = bbox3d
            new_detections[-1].distance = distance

        return new_detections

    def convert_bb_to_3d(
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        detection: Detection,
    ) -> BoundingBox3D | None:
        if depth_image is None or depth_image.size == 0:
            return None

        color_cx = float(detection.bbox.center.position.x)
        color_cy = float(detection.bbox.center.position.y)
        color_w = float(detection.bbox.size.x)
        color_h = float(detection.bbox.size.y)

        sx, sy = self._get_color_to_depth_scale(depth_image)
        depth_cx = color_cx * sx
        depth_cy = color_cy * sy
        depth_w = color_w * sx
        depth_h = color_h * sy

        img_h, img_w = depth_image.shape[:2]
        u_center = max(0, min(img_w - 1, int(round(depth_cx))))
        v_center = max(0, min(img_h - 1, int(round(depth_cy))))

        size_x = int(max(2, round(depth_w)))
        size_y = int(max(2, round(depth_h)))
        half_w = max(1, size_x // 5)
        half_h = max(1, size_y // 5)

        u_min = max(0, u_center - half_w)
        u_max = min(img_w - 1, u_center + half_w)
        v_min = max(0, v_center - half_h)
        v_max = min(img_h - 1, v_center + half_h)

        center_roi = depth_image[v_min : v_max + 1, u_min : u_max + 1].astype(np.float32)
        valid_depths = center_roi[np.isfinite(center_roi) & (center_roi > 0)]
        if valid_depths.size == 0:
            return None

        z_raw = float(np.percentile(valid_depths, 10))
        z = z_raw / float(self.depth_image_units_divisor)
        if not np.isfinite(z) or z <= 0.0:
            return None

        k = depth_info.k
        px, py, fx, fy = float(k[2]), float(k[5]), float(k[0]), float(k[4])
        if fx == 0.0 or fy == 0.0:
            return None

        cam_x = z * (float(u_center) - px) / fx
        cam_y = z * (float(v_center) - py) / fy
        cam_z = z

        robot_x = cam_z
        robot_y = cam_x
        robot_z = -cam_y

        msg = BoundingBox3D()
        msg.center.position.x = float(robot_x)
        msg.center.position.y = float(robot_y)
        msg.center.position.z = float(robot_z)

        w = z * (float(depth_w) / fx)
        h = z * (float(depth_h) / fy)
        msg.size.x = float(max(0.01, min(abs(w), abs(h))))
        msg.size.y = float(max(0.0, w))
        msg.size.z = float(max(0.0, h))
        return msg

    @staticmethod
    def _norm_name(name: str) -> str:
        return "".join([c for c in (name or "").lower() if c.isalnum()])

    def _update_cache(self, det_array_msg: DetectionArray) -> None:
        now = time.time()
        for det in det_array_msg.detections:
            if not det.class_name or not det.bbox3d.frame_id:
                continue
            key = self._norm_name(det.class_name)
            x = float(det.bbox3d.center.position.x)
            y = float(det.bbox3d.center.position.y)
            z = float(det.bbox3d.center.position.z)
            frame_id = str(det.bbox3d.frame_id)
            score = float(det.score)
            prev = self._target_cache.get(key)
            if prev is None or score >= prev[5]:
                self._target_cache[key] = (x, y, z, frame_id, now, score)

    def _publish_object_tf(self, det_array_msg: DetectionArray) -> None:
        if not det_array_msg.detections:
            return

        current_time = self.get_clock().now()
        for det in det_array_msg.detections:
            if not det.class_name or not det.bbox3d.frame_id:
                continue

            transform = TransformStamped()
            transform.header.stamp = current_time.to_msg()
            transform.header.frame_id = str(det.bbox3d.frame_id)
            transform.child_frame_id = str(det.class_name)
            transform.transform.translation.x = float(det.bbox3d.center.position.x)
            transform.transform.translation.y = float(det.bbox3d.center.position.y)
            transform.transform.translation.z = float(det.bbox3d.center.position.z)
            transform.transform.rotation.w = 1.0
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0
            self.tf_broadcaster.sendTransform(transform)

    def on_service(self, request, response):
        response.x = 0.0
        response.y = 0.0
        response.z = 0.0
        response.distance = 0.0
        response.frame_id = ""
        response.success = False

        try:
            query = ""
            if hasattr(request, "target_name"):
                query = request.target_name.strip()
            elif hasattr(request, "class_name"):
                query = request.class_name.strip()

            key_q = self._norm_name(query)
            if not key_q:
                self.get_logger().warn("[Service] Request name is empty.")
                return response

            now = time.time()
            stale = [
                k for k, v in self._target_cache.items()
                if (now - v[4]) > self.cache_timeout_sec
            ]
            for key in stale:
                del self._target_cache[key]

            found = self._target_cache.get(key_q)
            if found is None and self.match_substring:
                for key, value in self._target_cache.items():
                    if key_q in key or key in key_q:
                        found = value
                        break

            if found is None:
                self.get_logger().info(
                    f"[Service] Target '{query}' not found in cache. Current cache keys: {list(self._target_cache.keys())}"
                )
                return response

            x, y, z, frame_id, _, _ = found
            response.x = float(x)
            response.y = -float(y)
            response.z = float(z)
            response.distance = float(
                np.sqrt(
                    response.x * response.x
                    + response.y * response.y
                    + response.z * response.z
                )
            )
            response.frame_id = str(frame_id)
            response.success = True
            self.get_logger().info(
                f"[Service] Sent position for '{query}': X={response.x:.2f}, Y={response.y:.2f}, Z={response.z:.2f}, distance={response.distance:.2f}"
            )
            return response

        except Exception as exc:
            self.get_logger().error(f"[Service] GetTargetPosition error: {exc}")
            return response

    def get_transform(self, frame_id: str) -> Tuple[np.ndarray, np.ndarray] | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                frame_id,
                rclpy.time.Time(),
            )
            translation = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ]
            )
            rotation = np.array(
                [
                    transform.transform.rotation.w,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                ]
            )
            return translation, rotation
        except TransformException as exc:
            self.get_logger().error(f"Could not transform: {exc}")
            return None

    @staticmethod
    def transform_3d_box(
        bbox: BoundingBox3D,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> BoundingBox3D:
        position = (
            Detect3DNode.qv_mult(
                rotation,
                np.array(
                    [
                        bbox.center.position.x,
                        bbox.center.position.y,
                        bbox.center.position.z,
                    ]
                ),
            )
            + translation
        )

        bbox.center.position.x = position[0]
        bbox.center.position.y = position[1]
        bbox.center.position.z = position[2]

        size = Detect3DNode.qv_mult(
            rotation,
            np.array([bbox.size.x, bbox.size.y, bbox.size.z]),
        )
        bbox.size.x = abs(size[0])
        bbox.size.y = abs(size[1])
        bbox.size.z = abs(size[2])
        return bbox

    @staticmethod
    def qv_mult(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = np.array(q, dtype=np.float64)
        v = np.array(v, dtype=np.float64)
        qvec = q[1:]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2 * (uv * q[0] + uuv)


def main():
    rclpy.init()
    node = Detect3DNode()
    node.trigger_configure()
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
