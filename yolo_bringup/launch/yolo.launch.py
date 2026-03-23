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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        ("model_type", "YOLO", "Model type from Ultralytics (YOLO, World, YOLOE)"),
        ("model", "best.pt", "Model name or path"),
        ("device", "cuda:0", "Device to use (GPU/CPU)"),
        ("fuse_model", "False", "Whether to fuse the model for inference optimization"),
        ("yolo_encoding", "bgr8", "Encoding of the input image topic"),
        ("enable", "True", "Whether YOLO should run"),
        ("threshold", "0.5", "Minimum probability of a detection to be published"),
        ("iou", "0.7", "IoU threshold"),
        ("imgsz_height", "480", "Image height for inference"),
        ("imgsz_width", "640", "Image width for inference"),
        ("half", "False", "Whether to enable half-precision inference"),
        ("max_det", "300", "Maximum number of detections allowed per image"),
        ("augment", "False", "Whether to enable test-time augmentation"),
        ("agnostic_nms", "False", "Whether to enable class-agnostic NMS"),
        ("input_image_topic", "/camera/color/image_raw", "Input RGB image topic"),
        ("color_info_topic", "/camera/color/camera_info", "Input RGB camera info topic"),
        ("detections_topic", "detections", "Output 2D detections topic"),
        ("image_reliability", "1", "QoS reliability for input image topic"),
        ("use_3d", "True", "Whether to activate 3D detections"),
        ("input_depth_topic", "/camera/depth/image_raw", "Input depth image topic"),
        ("input_depth_info_topic", "/camera/depth/camera_info", "Input depth camera info topic"),
        ("detections_3d_topic", "detections_3d", "Output 3D detections topic"),
        ("depth_image_reliability", "1", "QoS reliability for depth image topic"),
        ("depth_info_reliability", "1", "QoS reliability for depth info topic"),
        ("target_frame", "base_link", "Target frame for 3D detections"),
        ("depth_image_units_divisor", "1000", "Divisor used to convert raw depth values into metres"),
        ("service_name", "get_target_position", "Target position service name"),
        ("namespace", "yolo", "Namespace for the nodes"),
    ]

    declared_args = [
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in args
    ]

    yolo_node = Node(
        package="yolo_ros",
        executable="yolo_node",
        name="yolo_node",
        namespace=LaunchConfiguration("namespace"),
        parameters=[
            {
                "model_type": LaunchConfiguration("model_type"),
                "model": LaunchConfiguration("model"),
                "device": LaunchConfiguration("device"),
                "fuse_model": LaunchConfiguration("fuse_model"),
                "yolo_encoding": LaunchConfiguration("yolo_encoding"),
                "enable": LaunchConfiguration("enable"),
                "threshold": LaunchConfiguration("threshold"),
                "iou": LaunchConfiguration("iou"),
                "imgsz_height": LaunchConfiguration("imgsz_height"),
                "imgsz_width": LaunchConfiguration("imgsz_width"),
                "half": LaunchConfiguration("half"),
                "max_det": LaunchConfiguration("max_det"),
                "augment": LaunchConfiguration("augment"),
                "agnostic_nms": LaunchConfiguration("agnostic_nms"),
                "input_image_topic": LaunchConfiguration("input_image_topic"),
                "detections_topic": LaunchConfiguration("detections_topic"),
                "image_reliability": LaunchConfiguration("image_reliability"),
            }
        ],
    )

    detect_3d_node = Node(
        package="yolo_ros",
        executable="detect_3d_node",
        name="detect_3d_node",
        namespace=LaunchConfiguration("namespace"),
        condition=IfCondition(LaunchConfiguration("use_3d")),
        parameters=[
            {
                "color_info_topic": LaunchConfiguration("color_info_topic"),
                "depth_image_topic": LaunchConfiguration("input_depth_topic"),
                "depth_info_topic": LaunchConfiguration("input_depth_info_topic"),
                "detections_topic": LaunchConfiguration("detections_topic"),
                "detections_3d_topic": LaunchConfiguration("detections_3d_topic"),
                "target_frame": LaunchConfiguration("target_frame"),
                "depth_image_units_divisor": LaunchConfiguration("depth_image_units_divisor"),
                "depth_image_reliability": LaunchConfiguration("depth_image_reliability"),
                "depth_info_reliability": LaunchConfiguration("depth_info_reliability"),
                "service_name": LaunchConfiguration("service_name"),
            }
        ],
    )

    return LaunchDescription(declared_args + [yolo_node, detect_3d_node])
