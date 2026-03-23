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

from typing import Dict, List

from cv_bridge import CvBridge

import rclpy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

import torch
from sensor_msgs.msg import Image
from ultralytics import YOLO, YOLOE, YOLOWorld
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Results
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray


class YoloNode(LifecycleNode):
    """ROS 2 Lifecycle Node for parameter-driven YOLO object detection."""

    def __init__(self) -> None:
        super().__init__("yolo_node")

        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("model", "best.pt")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("fuse_model", False)
        self.declare_parameter("yolo_encoding", "bgr8")
        self.declare_parameter("enable", True)
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("augment", False)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("input_image_topic", "image_raw")
        self.declare_parameter("detections_topic", "detections")
        self.declare_parameter("world_classes", [])

        self.type_to_model = {"YOLO": YOLO, "World": YOLOWorld, "YOLOE": YOLOE}

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.model_type = self.get_parameter("model_type").get_parameter_value().string_value
        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.fuse_model = self.get_parameter("fuse_model").get_parameter_value().bool_value
        self.yolo_encoding = self.get_parameter("yolo_encoding").get_parameter_value().string_value
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = self.get_parameter("imgsz_height").get_parameter_value().integer_value
        self.imgsz_width = self.get_parameter("imgsz_width").get_parameter_value().integer_value
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.augment = self.get_parameter("augment").get_parameter_value().bool_value
        self.agnostic_nms = self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        self.input_image_topic = self.get_parameter("input_image_topic").get_parameter_value().string_value
        self.detections_topic = self.get_parameter("detections_topic").get_parameter_value().string_value
        self.world_classes = list(self.get_parameter("world_classes").get_parameter_value().string_array_value)
        reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value

        self.image_qos_profile = QoSProfile(
            reliability=reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self._pub = self.create_lifecycle_publisher(DetectionArray, self.detections_topic, 10)
        self.cv_bridge = CvBridge()
        self._sub = None
        self.yolo = None

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        try:
            self.yolo = self.type_to_model[self.model_type](self.model)
            self.yolo.to(self.device)
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exist")
            return TransitionCallbackReturn.ERROR
        except Exception as exc:
            self.get_logger().error(f"Failed to load model '{self.model}': {exc}")
            return TransitionCallbackReturn.ERROR

        if self.fuse_model and isinstance(self.yolo, (YOLO, YOLOWorld)):
            try:
                self.yolo.fuse()
            except TypeError as exc:
                self.get_logger().warn(f"Error while fusing model: {exc}")

        if isinstance(self.yolo, YOLOWorld) and self.world_classes:
            self.yolo.set_classes(self.world_classes)
            self.get_logger().info(f"Using world classes: {self.world_classes}")

        self._sub = self.create_subscription(
            Image,
            self.input_image_topic,
            self.image_cb,
            self.image_qos_profile,
        )

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None

        if self.yolo is not None:
            del self.yolo
            self.yolo = None
            if "cuda" in self.device:
                torch.cuda.empty_cache()

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")
        self.destroy_publisher(self._pub)
        del self.image_qos_profile
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_shutdown(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def parse_hypothesis(self, results: Results) -> List[Dict]:
        hypothesis_list: List[Dict] = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                hypothesis_list.append(
                    {
                        "class_id": int(box_data.cls),
                        "class_name": self.yolo.names[int(box_data.cls)],
                        "score": float(box_data.conf),
                    }
                )
        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                hypothesis_list.append(
                    {
                        "class_id": int(results.obb.cls[i]),
                        "class_name": self.yolo.names[int(results.obb.cls[i])],
                        "score": float(results.obb.conf[i]),
                    }
                )

        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:
        boxes_list: List[BoundingBox2D] = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                msg = BoundingBox2D()
                box = box_data.xywh[0]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])
                boxes_list.append(msg)
        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                msg = BoundingBox2D()
                box = results.obb.xywhr[i]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.center.theta = float(box[4])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])
                boxes_list.append(msg)

        return boxes_list

    def image_cb(self, msg: Image) -> None:
        if not self.get_parameter("enable").get_parameter_value().bool_value:
            return

        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding=self.yolo_encoding)
        results = self.yolo.predict(
            source=cv_image,
            verbose=False,
            stream=False,
            conf=self.threshold,
            iou=self.iou,
            imgsz=(self.imgsz_height, self.imgsz_width),
            half=self.half,
            max_det=self.max_det,
            augment=self.augment,
            agnostic_nms=self.agnostic_nms,
            device=self.device,
        )
        results = results[0].cpu()

        detections_msg = DetectionArray()
        detections_msg.header = msg.header

        if results.boxes or results.obb:
            hypothesis = self.parse_hypothesis(results)
            boxes = self.parse_boxes(results)

            for det_hypothesis, box in zip(hypothesis, boxes):
                aux_msg = Detection()
                aux_msg.class_id = det_hypothesis["class_id"]
                aux_msg.class_name = det_hypothesis["class_name"]
                aux_msg.score = det_hypothesis["score"]
                aux_msg.bbox = box
                detections_msg.detections.append(aux_msg)

        self._pub.publish(detections_msg)

        del results
        del cv_image


def main():
    rclpy.init()
    node = YoloNode()
    node.trigger_configure()
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
