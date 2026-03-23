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
import numpy as np
from typing import List, Tuple, Optional

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

import torch
from ultralytics import YOLO, YOLOWorld, YOLOE
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes

from sensor_msgs.msg import CameraInfo, Image
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.msg import BoundingBox3D
from yolo_msgs.srv import GetTargetPosition


class IntegratedDetectionNode(LifecycleNode):
    """
    통합 객체 탐지 노드
    
    서비스 요청을 받으면 YOLO 추론을 수행하고 3D position까지 계산하여 반환합니다.
    """

    def __init__(self) -> None:
        """
        노드 초기화
        """
        super().__init__("integrated_detection_node")

        # YOLO 파라미터
        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("model", "best.pt")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("fuse_model", False)
        self.declare_parameter("yolo_encoding", "bgr8")
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("augment", False)
        self.declare_parameter("agnostic_nms", False)

        # 3D Detection 파라미터
        self.declare_parameter("target_frame", "arm_link")
        self.declare_parameter("depth_image_units_divisor", 1000)
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_image_topic", "/camera/depth/image_raw")
        self.declare_parameter("depth_info_topic", "/camera/depth/camera_info")
        self.declare_parameter("color_info_topic", "/camera/color/camera_info")
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("service_name", "get_target_position")
        self.declare_parameter("match_substring", True)
        self.declare_parameter("use_tf", True)

        self.type_to_model = {"YOLO": YOLO, "World": YOLOWorld, "YOLOE": YOLOE}
        
        # 최신 메시지 저장
        self._latest_image_msg = None
        self._latest_depth_msg = None
        self._latest_depth_info_msg = None
        self._color_info = None
        
        # 유틸리티
        self.cv_bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.yolo = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Configure lifecycle callback
        """
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        # YOLO 파라미터 로드
        self.model_type = (
            self.get_parameter("model_type").get_parameter_value().string_value
        )
        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.fuse_model = (
            self.get_parameter("fuse_model").get_parameter_value().bool_value
        )
        self.yolo_encoding = (
            self.get_parameter("yolo_encoding").get_parameter_value().string_value
        )
        self.threshold = (
            self.get_parameter("threshold").get_parameter_value().double_value
        )
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = (
            self.get_parameter("imgsz_height").get_parameter_value().integer_value
        )
        self.imgsz_width = (
            self.get_parameter("imgsz_width").get_parameter_value().integer_value
        )
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.augment = self.get_parameter("augment").get_parameter_value().bool_value
        self.agnostic_nms = (
            self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        )

        # 3D Detection 파라미터 로드
        self.target_frame = (
            self.get_parameter("target_frame").get_parameter_value().string_value
        )
        self.depth_image_units_divisor = (
            self.get_parameter("depth_image_units_divisor")
            .get_parameter_value()
            .integer_value
        )

        # QoS 프로파일 설정
        reliability = (
            self.get_parameter("image_reliability")
            .get_parameter_value()
            .integer_value
        )
        self.image_qos_profile = QoSProfile(
            reliability=reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        dimg_reliability = (
            self.get_parameter("depth_image_reliability")
            .get_parameter_value()
            .integer_value
        )
        self.depth_image_qos_profile = QoSProfile(
            reliability=dimg_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        dinfo_reliability = (
            self.get_parameter("depth_info_reliability")
            .get_parameter_value()
            .integer_value
        )
        self.depth_info_qos_profile = QoSProfile(
            reliability=dinfo_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # TF 리스너 초기화
        self.tf_listener = TransformListener(self.tf_buffer, self)

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Activate lifecycle callback
        """
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        # YOLO 모델 로드
        try:
            self.yolo = self.type_to_model[self.model_type](self.model)
            self.yolo.to(self.device)
            self.get_logger().info(f"YOLO model loaded: {self.model}")
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exists")
            return TransitionCallbackReturn.ERROR
        except Exception as e:
            self.get_logger().error(f"Error loading model: {e}")
            return TransitionCallbackReturn.ERROR

        # YOLO 모델 퓨즈 (지원되는 경우)
        if self.fuse_model and (
            isinstance(self.yolo, YOLO) or isinstance(self.yolo, YOLOWorld)
        ):
            try:
                self.get_logger().info("Trying to fuse model...")
                self.yolo.fuse()
            except TypeError as e:
                self.get_logger().warn(f"Error while fuse: {e}")

        # 서비스 생성
        srv_name = self.get_parameter("service_name").get_parameter_value().string_value
        self._srv = self.create_service(GetTargetPosition, srv_name, self.on_service)
        self.get_logger().info(f"GetTargetPosition service ready: {srv_name}")

        # 구독 생성
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        depth_image_topic = self.get_parameter("depth_image_topic").get_parameter_value().string_value
        depth_info_topic = self.get_parameter("depth_info_topic").get_parameter_value().string_value
        color_info_topic = self.get_parameter("color_info_topic").get_parameter_value().string_value

        self._sub_image = self.create_subscription(
            Image, image_topic, self._on_image, self.image_qos_profile
        )
        self._sub_depth = self.create_subscription(
            Image, depth_image_topic, self._on_depth, self.depth_image_qos_profile
        )
        self._sub_depth_info = self.create_subscription(
            CameraInfo, depth_info_topic, self._on_depth_info, self.depth_info_qos_profile
        )
        self._sub_color_info = self.create_subscription(
            CameraInfo, color_info_topic, self._on_color_info, 10
        )

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Deactivate lifecycle callback
        """
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        if self.yolo is not None:
            del self.yolo
            if "cuda" in self.device:
                self.get_logger().info("Clearing CUDA cache")
                torch.cuda.empty_cache()

        self.destroy_service(self._srv)
        self.destroy_subscription(self._sub_image)
        self.destroy_subscription(self._sub_depth)
        self.destroy_subscription(self._sub_depth_info)
        self.destroy_subscription(self._sub_color_info)

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Cleanup lifecycle callback
        """
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        del self.tf_listener

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Shutdown lifecycle callback
        """
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_shutdown(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def _on_image(self, msg: Image) -> None:
        """이미지 콜백"""
        self._latest_image_msg = msg

    def _on_depth(self, msg: Image) -> None:
        """깊이 이미지 콜백"""
        self._latest_depth_msg = msg

    def _on_depth_info(self, msg: CameraInfo) -> None:
        """깊이 카메라 정보 콜백"""
        self._latest_depth_info_msg = msg

    def _on_color_info(self, msg: CameraInfo) -> None:
        """컬러 카메라 정보 콜백"""
        self._color_info = msg

    def _get_color_to_depth_scale(self, depth_image: np.ndarray):
        """
        컬러 이미지에서 깊이 이미지로의 좌표 스케일 반환
        """
        if self._color_info is None:
            return 1.0, 1.0

        color_w = int(self._color_info.width)
        color_h = int(self._color_info.height)

        if color_w <= 0 or color_h <= 0:
            return 1.0, 1.0

        depth_h, depth_w = depth_image.shape[:2]

        if color_w == depth_w and color_h == depth_h:
            return 1.0, 1.0

        sx = depth_w / float(color_w)
        sy = depth_h / float(color_h)
        return sx, sy

    def _norm_name(self, name: str) -> str:
        """클래스 이름 정규화"""
        name = name or ""
        return "".join([c for c in name.lower() if c.isalnum()])

    def on_service(self, request, response):
        """
        서비스 콜백: 추론 수행 및 3D position 반환
        """
        # 기본값 설정
        response.x = 0.0
        response.y = 0.0
        response.z = 0.0
        response.distance = 0.0
        response.frame_id = ""
        response.success = False

        try:
            # 1. 요청 파라미터 확인
            query = ""
            if hasattr(request, 'target_name'):
                query = request.target_name.strip()
            elif hasattr(request, 'class_name'):
                query = request.class_name.strip()

            if not query:
                self.get_logger().warn("[Service] Request name is empty.")
                return response

            # 2. 최신 이미지 확인
            if self._latest_image_msg is None:
                self.get_logger().warn("[Service] No image available.")
                return response

            if self._latest_depth_msg is None:
                self.get_logger().warn("[Service] No depth image available.")
                return response

            if self._latest_depth_info_msg is None:
                self.get_logger().warn("[Service] No depth info available.")
                return response

            # 3. YOLO 추론 수행
            if self.yolo is None:
                self.get_logger().error("[Service] YOLO model not loaded.")
                return response

            self.get_logger().info(f"[Service] Running inference for '{query}'...")
            cv_image = self.cv_bridge.imgmsg_to_cv2(
                self._latest_image_msg, desired_encoding=self.yolo_encoding
            )

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
            results: Results = results[0].cpu()

            # 4. 탐지 결과 파싱
            detections = self._parse_detections(results)
            if not detections:
                self.get_logger().info(f"[Service] No detections found.")
                return response

            # 5. 요청한 클래스 찾기
            key_q = self._norm_name(query)
            found_detection = None

            for det in detections:
                det_key = self._norm_name(det.class_name)
                if det_key == key_q:
                    found_detection = det
                    break

            # substring 매칭
            if found_detection is None and bool(self.get_parameter("match_substring").value):
                for det in detections:
                    det_key = self._norm_name(det.class_name)
                    if key_q in det_key or det_key in key_q:
                        found_detection = det
                        break

            if found_detection is None:
                detected_classes = [self._norm_name(d.class_name) for d in detections]
                self.get_logger().info(
                    f"[Service] Target '{query}' not found. Detected classes: {detected_classes}"
                )
                return response

            # 6. 3D position 계산
            self.get_logger().info(f"[Service] Computing 3D position for '{found_detection.class_name}'...")
            bbox3d = self._convert_bb_to_3d(
                self._latest_depth_msg,
                self._latest_depth_info_msg,
                found_detection
            )

            if bbox3d is None:
                self.get_logger().warn("[Service] Failed to compute 3D position.")
                return response

            # 7. TF 변환 (필요한 경우)
            if bool(self.get_parameter("use_tf").value):
                transform = self._get_transform(self._latest_depth_info_msg.header.frame_id)
                if transform is not None:
                    bbox3d = self._transform_3d_box(bbox3d, transform[0], transform[1])
                    bbox3d.frame_id = self.target_frame
                else:
                    bbox3d.frame_id = self._latest_depth_info_msg.header.frame_id
            else:
                bbox3d.frame_id = self._latest_depth_info_msg.header.frame_id

            # 8. 결과 반환
            response.x = float(bbox3d.center.position.x)
            response.y = float(bbox3d.center.position.y)
            response.z = float(bbox3d.center.position.z)
            response.distance = float(
                np.sqrt(
                    response.x * response.x
                    + response.y * response.y
                    + response.z * response.z
                )
            )
            response.frame_id = str(bbox3d.frame_id)
            response.success = True

            self.get_logger().info(
                f"[Service] Success! Position for '{query}': "
                f"X={response.x:.3f}, Y={response.y:.3f}, Z={response.z:.3f}, "
                f"distance={response.distance:.3f} "
                f"(frame: {response.frame_id})"
            )

            return response

        except Exception as e:
            self.get_logger().error(f"[Service] Error: {e}", exc_info=True)
            return response

    def _parse_detections(self, results: Results) -> List[Detection]:
        """
        YOLO 결과를 Detection 메시지 리스트로 변환
        """
        detections = []

        if not (results.boxes or results.obb):
            return detections

        # Hypothesis 파싱
        if self.yolo is None:
            return detections

        hypothesis_list = []
        if results.boxes:
            for box_data in results.boxes:
                hypothesis = {
                    "class_id": int(box_data.cls),
                    "class_name": self.yolo.names[int(box_data.cls)],
                    "score": float(box_data.conf),
                }
                hypothesis_list.append(hypothesis)
        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                hypothesis = {
                    "class_id": int(results.obb.cls[i]),
                    "class_name": self.yolo.names[int(results.obb.cls[i])],
                    "score": float(results.obb.conf[i]),
                }
                hypothesis_list.append(hypothesis)

        # Bounding boxes 파싱
        from yolo_msgs.msg import BoundingBox2D
        boxes_list = []
        if results.boxes:
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

        # Detection 메시지 생성
        for i in range(len(hypothesis_list)):
            det = Detection()
            det.class_id = hypothesis_list[i]["class_id"]
            det.class_name = hypothesis_list[i]["class_name"]
            det.score = hypothesis_list[i]["score"]
            det.bbox = boxes_list[i]
            detections.append(det)

        return detections
    def _convert_bb_to_3d(
        self, depth_msg: Image, depth_info: CameraInfo, detection: Detection
    ) -> Optional[BoundingBox3D]:
        """
        2D 바운딩 박스를 3D로 변환 - 중심 주변에서 유효한 깊이 값 찾기
        """
        try:
            depth_image = self.cv_bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )

            if depth_image is None or depth_image.size == 0:
                self.get_logger().warn("[3D] Depth image is None or empty")
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
            u_center = int(round(depth_cx))
            v_center = int(round(depth_cy))
            u_center = max(0, min(img_w - 1, u_center))
            v_center = max(0, min(img_h - 1, v_center))

            # 중심 주변에서 유효한 깊이 값 찾기 (스파이럴 탐색)
            max_radius = min(int(depth_w / 2), int(depth_h / 2), 50)  # 최대 탐색 반경 제한
            found_u, found_v, z_raw = None, None, None
            
            # 중심부터 시작해서 주변을 탐색
            for radius in range(max_radius + 1):
                if radius == 0:
                    # 중심 점 확인
                    depth_val = float(depth_image[v_center, u_center])
                    if np.isfinite(depth_val) and depth_val > 0:
                        found_u, found_v = u_center, v_center
                        z_raw = depth_val
                        break
                else:
                    # 반경 radius인 원 주변 탐색
                    found = False
                    for du in range(-radius, radius + 1):
                        for dv in range(-radius, radius + 1):
                            # 원의 경계만 확인 (내부는 이미 확인됨)
                            if abs(du) != radius and abs(dv) != radius:
                                continue
                            
                            u = u_center + du
                            v = v_center + dv
                            
                            # 이미지 범위 체크
                            if u < 0 or u >= img_w or v < 0 or v >= img_h:
                                continue
                            
                            depth_val = float(depth_image[v, u])
                            if np.isfinite(depth_val) and depth_val > 0:
                                found_u, found_v = u, v
                                z_raw = depth_val
                                found = True
                                break
                        
                        if found:
                            break
                    
                    if found:
                        break
            
            # 유효한 깊이 값을 찾지 못한 경우
            if z_raw is None or z_raw <= 0:
                self.get_logger().warn(
                    f"[3D] No valid depth found around center ({u_center}, {v_center}) "
                    f"within radius {max_radius}"
                )
                return None

            z = z_raw / float(self.depth_image_units_divisor)
            if not np.isfinite(z) or z <= 0.0:
                self.get_logger().warn(
                    f"[3D] Invalid depth after conversion: z={z}, z_raw={z_raw}, "
                    f"divisor={self.depth_image_units_divisor}"
                )
                return None

            k = depth_info.k
            px, py, fx, fy = float(k[2]), float(k[5]), float(k[0]), float(k[4])
            if fx == 0.0 or fy == 0.0:
                self.get_logger().warn(
                    f"[3D] Invalid camera parameters: fx={fx}, fy={fy}"
                )
                return None

            # 찾은 좌표로 카메라 좌표계 계산
            cam_x = z * (float(found_u) - px) / fx
            cam_y = z * (float(found_v) - py) / fy
            cam_z = z

            # 로봇 좌표계로 변환
            robot_x = cam_z
            robot_y = cam_x
            robot_z = -cam_y

            msg = BoundingBox3D()
            msg.center.position.x = float(robot_x)
            msg.center.position.y = float(robot_y)
            msg.center.position.z = float(robot_z)

            # 크기 변환
            w = z * (float(depth_w) / fx)
            h = z * (float(depth_h) / fy)
            msg.size.x = float(max(0.01, min(abs(w), abs(h))))
            msg.size.y = float(max(0.0, w))
            msg.size.z = float(max(0.0, h))

            self.get_logger().debug(
                f"[3D] Success: position=({robot_x:.3f}, {robot_y:.3f}, {robot_z:.3f}), "
                f"depth={z:.3f}m at ({found_u}, {found_v}), "
                f"searched from center ({u_center}, {v_center})"
            )

            return msg

        except Exception as e:
            self.get_logger().error(f"[3D] Error converting to 3D: {e}")
            return None

    def _get_transform(self, frame_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        TF 변환 가져오기
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, frame_id, rclpy.time.Time()
            )

            translation = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ])

            rotation = np.array([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
            ])

            return translation, rotation

        except TransformException as ex:
            self.get_logger().warn(f"Could not transform: {ex}")
            return None

    @staticmethod
    def _transform_3d_box(
        bbox: BoundingBox3D,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> BoundingBox3D:
        """
        3D 바운딩 박스 변환
        """
        # Quaternion-vector multiplication
        def qv_mult(q: np.ndarray, v: np.ndarray) -> np.ndarray:
            q = np.array(q, dtype=np.float64)
            v = np.array(v, dtype=np.float64)
            qvec = q[1:]
            uv = np.cross(qvec, v)
            uuv = np.cross(qvec, uv)
            return v + 2 * (uv * q[0] + uuv)

        # Position 변환
        position = qv_mult(
            rotation,
            np.array([
                bbox.center.position.x,
                bbox.center.position.y,
                bbox.center.position.z,
            ])
        ) + translation

        bbox.center.position.x = position[0]
        bbox.center.position.y = position[1]
        bbox.center.position.z = position[2]

        # Size 변환 (rotation만)
        size = qv_mult(rotation, np.array([bbox.size.x, bbox.size.y, bbox.size.z]))
        bbox.size.x = abs(size[0])
        bbox.size.y = abs(size[1])
        bbox.size.z = abs(size[2])

        return bbox

def main():
    rclpy.init()
    node = IntegratedDetectionNode()
    node.trigger_configure()
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
