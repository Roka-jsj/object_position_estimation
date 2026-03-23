#!/usr/bin/env python3
import sys

import rclpy
from rclpy.node import Node
from yolo_msgs.srv import GetTargetPosition


class PositionClient(Node):
    def __init__(self):
        super().__init__("position_client")
        self.declare_parameter("service_namespace", "yolo")
        self.declare_parameter("service_name", "get_target_position")
        self.declare_parameter("wait_timeout_sec", 1.0)
        self.declare_parameter("target_name", "")

        namespace = self.get_parameter("service_namespace").get_parameter_value().string_value.strip("/")
        service_name = self.get_parameter("service_name").get_parameter_value().string_value.strip()
        self.wait_timeout_sec = self.get_parameter("wait_timeout_sec").get_parameter_value().double_value
        self.default_target_name = self.get_parameter("target_name").get_parameter_value().string_value.strip()

        if service_name.startswith("/"):
            self.service_fqn = service_name
        elif namespace:
            self.service_fqn = f"/{namespace}/{service_name}"
        else:
            self.service_fqn = service_name

        self.cli = self.create_client(GetTargetPosition, self.service_fqn)

    def call(self, name: str):
        while not self.cli.wait_for_service(timeout_sec=self.wait_timeout_sec):
            self.get_logger().info(f"waiting for service {self.service_fqn}...")

        req = GetTargetPosition.Request()
        req.class_name = name

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def main():
    rclpy.init()
    node = PositionClient()

    target = sys.argv[1] if len(sys.argv) >= 2 else node.default_target_name
    if not target:
        print(
            "Usage: ros2 run yolo_ros position <class_name> --ros-args -p service_namespace:=yolo -p service_name:=get_target_position"
        )
        node.destroy_node()
        rclpy.shutdown()
        return

    res = node.call(target)
    node.destroy_node()
    rclpy.shutdown()

    if res is None or not res.success:
        print("No detection found.")
        return

    print(f"class: {target}")
    print(f"x: {res.x:.2f}")
    print(f"y: {res.y:.2f}")
    print(f"z: {res.z:.2f}")
    print(f"distance: {res.distance:.2f}")
    print(f"frame: {res.frame_id}")


if __name__ == "__main__":
    main()
