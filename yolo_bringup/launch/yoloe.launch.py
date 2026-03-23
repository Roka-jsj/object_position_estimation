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

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("yolo_bringup"),
                        "launch",
                        "yolo.launch.py",
                    )
                ),
                launch_arguments={
                    "model_type": "YOLOE",
                    "model": LaunchConfiguration("model", default="yoloe-11l-seg-pf.pt"),
                    "device": LaunchConfiguration("device", default="cuda:0"),
                    "enable": LaunchConfiguration("enable", default="True"),
                    "threshold": LaunchConfiguration("threshold", default="0.5"),
                    "input_image_topic": LaunchConfiguration("input_image_topic", default="/camera/rgb/image_raw"),
                    "color_info_topic": LaunchConfiguration("color_info_topic", default="/camera/rgb/camera_info"),
                    "input_depth_topic": LaunchConfiguration("input_depth_topic", default="/camera/depth/image_raw"),
                    "input_depth_info_topic": LaunchConfiguration("input_depth_info_topic", default="/camera/depth/camera_info"),
                    "image_reliability": LaunchConfiguration("image_reliability", default="1"),
                    "depth_image_reliability": LaunchConfiguration("depth_image_reliability", default="1"),
                    "depth_info_reliability": LaunchConfiguration("depth_info_reliability", default="1"),
                    "target_frame": LaunchConfiguration("target_frame", default="base_link"),
                    "service_name": LaunchConfiguration("service_name", default="get_target_position"),
                    "namespace": LaunchConfiguration("namespace", default="yolo"),
                    "use_3d": LaunchConfiguration("use_3d", default="True"),
                }.items(),
            )
        ]
    )
