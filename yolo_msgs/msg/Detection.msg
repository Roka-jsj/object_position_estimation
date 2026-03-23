# defines a YOLO detection result

# class probability
int32 class_id
string class_name
float64 score

# 2D bounding box surrounding the object in pixels
yolo_msgs/BoundingBox2D bbox

# 3D bounding box surrounding the object in meters
yolo_msgs/BoundingBox3D bbox3d

# distance from the detection origin to the 3D box center in meters
float64 distance
