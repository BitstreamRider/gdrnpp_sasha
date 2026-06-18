# inference with detector, gdrn, and refiner
import os.path as osp
import sys
import threading

#from lib.vis_utils import image
#import torch
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)

#from predictor_gdrn import GdrnPredictor
import os
import argparse
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Header, Float32MultiArray
from geometry_msgs.msg import Pose, Point, Quaternion
from robokudo_msgs.action import GenericImgProcAnnotator
import json
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import tf_transformations
from geometry_msgs.msg import TransformStamped
#from lib.render_vispy.renderer import RendererROS
import queue

from threeD_boxes import THREED_BOXES
# adding subprocess for calling an older version of python to run gdrn
import subprocess
import json
import tempfile


class GDRN_ROS(Node):
    def __init__(self, renderer_request_queue, renderer_result_queue, dataset_name):
            super().__init__("gdrn_estimation")
            self.declare_parameter('/pose_estimator/intrinsics',  [554.3827, 0.0, 320.0, 0.0, 554.3827, 240.0, 0.0, 0.0, 1.0])
            self.declare_parameter('/pose_estimator/color_frame_id', "head_rgbd_sensor_rgb_frame")
            self.intrinsics = np.asarray(self.get_parameter('/pose_estimator/intrinsics').value).reshape(3, 3)
            self.frame_id = self.get_parameter('/pose_estimator/color_frame_id').value

            self.reflow = "tracebotcanister" in dataset_name
            self.config_path = osp.join(PROJ_ROOT,"configs/gdrn/" + dataset_name + "/" + dataset_name + "_inference.py")
            self.weights_path = osp.join(PROJ_ROOT,"output/gdrnpp_" + dataset_name + "_weights.pth")
            self.models_path = osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/" + dataset_name + "/models")
            self.dataset_name = dataset_name
            

            self.renderer_request_queue = renderer_request_queue
            self.renderer_result_queue = renderer_result_queue

            self.br = tf2_ros.TransformBroadcaster(self)
            self.server = ActionServer(self, GenericImgProcAnnotator, '/pose_estimator/gdrnet', self.estimate_pose)

            qos = QoSProfile(depth=1)
            qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            self.plane_normal = None
            self.plane_subscription = self.create_subscription(
                Float32MultiArray,
                '/grasping_pipeline/plane_normal',
                self.callback_plane_normal,
                qos
            )
            self.plane_point = None
            self.point_subscription = self.create_subscription(
                Point,
                '/grasping_pipeline/plane_point',
                self.callback_plane_point,
                qos
            )

            self.get_logger().info("GDRNPP ROS node initialized and action server started.")
            #print("Pose Estimation with GDRNPP is ready.")
    
    def callback_plane_normal(self, msg):
        self.get_logger().info(f"Received plane normal: {msg.data}")
        self.plane_normal = np.array(msg.data)
    
    def callback_plane_point(self, msg):
        self.get_logger().info(f"Received plane point: ({msg.x}, {msg.y}, {msg.z})")
        self.plane_point = np.array([msg.x, msg.y, msg.z])
    """
    When using the robokudo_msgs, as the callback function for the action server
    """
    def estimate_pose(self, goal_handle):
        #print("request detection...")
        start_time = time.time()

        # === IN ===
        # --- rgb
        goal = goal_handle.request
        bb_detections = goal.bb_detections
        class_names = goal.class_names
        description = goal.description
        scores = json.loads(description)
        rgb = goal.rgb
        depth = goal.depth

        width, height = rgb.width, rgb.height
        assert width == 640 and height == 480

        try:
            image = CvBridge().imgmsg_to_cv2(rgb, "bgr8")
        except CvBridgeError as e:
            print(e)

        if not self.reflow:
            try:
                depth.encoding = "mono16"
                depth_img = CvBridge().imgmsg_to_cv2(depth, "mono16")
                #depth_img = depth_img/1000 keeping it raw
            except CvBridgeError as e:
                print(e)
        else:
            depth_img = None
        
        fd, rgb_path = tempfile.mkstemp(suffix=".png")
        cv2.imwrite(rgb_path, image)
        depth_path = None
        if depth_img is not None:
            fd, depth_path = tempfile.mkstemp(suffix=".npy")
            np.save(depth_path, depth_img)
  
        valid_class_names = []
        pose_results = []
        class_confidences = []
        print(f"Received {len(bb_detections)} bounding box detections. Processing each detection...")
        for name, roi in zip(class_names, bb_detections):
            print(f"Processing detection for class: {name}")
            if name == "036_wood_block":
                continue
            score = np.float32(scores[name])

            
            ymin = roi.x_offset
            xmin = roi.y_offset
            ymax = ymin + roi.width
            xmax = xmin + roi.height

            if self.reflow:
                plane_normal = self.plane_normal
                plane_pt = self.plane_point

                print("=============================================\n\n\n\n")
                print(f"Plane normal: {plane_normal}")
                print(f"Plane point: {plane_pt}")
                print("\n\n\n\n=============================================")
            else:
                plane_normal = []
                plane_pt = []
            
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
                json.dump({
                    "image_path_rgb": rgb_path,
                    "image_path_depth": depth_path,
                    "ymin": ymin,
                    "xmin": xmin,
                    "ymax": ymax,
                    "xmax": xmax,
                    "name": name,
                    "score": float(score),
                    "reflow": self.reflow,
                    "plane_normal": plane_normal,
                    "plane_pt": plane_pt,
                    "intrinsics": self.intrinsics.tolist(),
                    "dataset": self.dataset_name,
                    "config": self.config_path,
                    "weights": self.weights_path,
                    "models": self.models_path
                }, f)
                input_file = f.name
            #self.gdrn_predictor.gdrn_visualization(batch=data_dict, out_dict=out_dict, image=image)
            proc = subprocess.run(["/opt/py38/bin/python", "gdrn.py", input_file], capture_output=True, text=True)
            # Optional: check for errors
            print("RETURN CODE:", proc.returncode)
            print("STDOUT:", repr(proc.stdout))
            print("STDERR:", repr(proc.stderr))

            # Parse JSON output
            data = json.loads(proc.stdout.strip().split("\n")[-1])

            obj_name = data["obj_names"]
            poses = data["poses"]
            pose = np.array(poses[obj_name])
            print(pose)
            #obj_name = self.gdrn_predictor.objs[int(obj_id)]

            R_0 = np.eye(4,4)
            R = pose[ 0:3,0:3 ]
            R_0[ 0:3,0:3 ] = R
            t = pose[ 0:3,3:4 ].ravel()

            ###########################################################
            if self.reflow is True:
                if len(plane_normal) != 0 and len(plane_pt) != 0:
                    ori_points = np.ascontiguousarray(THREED_BOXES["tracebotcanister"], dtype=np.float32)
                    oriented_bbox = pose[:3,:3].dot(ori_points.T).T
                    oriented_bbox += np.repeat(pose[:3,3][np.newaxis, :], 8, axis=0)  # * 0.001

                    # Choose the bounding box point that has the smallest dot product to the plane normal (signed)
                    min_dot_prod_to_plane_idx = np.argmin(np.dot((oriented_bbox - plane_pt), plane_normal))
                    min_dot_prod_to_plane_pt = oriented_bbox[min_dot_prod_to_plane_idx, :]

                    # Create the corresponding normalized ray.
                    ray = min_dot_prod_to_plane_pt / np.linalg.norm(min_dot_prod_to_plane_pt)

                    # Compute the corresponding intersection between plane and ray
                    t = np.dot(plane_normal, plane_pt) / np.dot(plane_normal, ray)
                    pt_ray_intersection = t*ray

                    pose[:3,3] += pt_ray_intersection - min_dot_prod_to_plane_pt
                else:
                    print("No plane normal or plane point received from ROS param server!")
            ###########################################################

            rot_quat = tf_transformations.quaternion_from_matrix(R_0)

            
            t = TransformStamped()

            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.frame_id

            t.child_frame_id = f"pose_{obj_name}"

            t.transform.translation.x = pose[0][3]
            t.transform.translation.y = pose[1][3]
            t.transform.translation.z = pose[2][3]

            t.transform.rotation.x = rot_quat[0]
            t.transform.rotation.y = rot_quat[1]
            t.transform.rotation.z = rot_quat[2]
            t.transform.rotation.w = rot_quat[3]

            self.br.sendTransform(t)

            confidence = float(score)
            final_pose = Pose()
            final_pose.position.x = pose[0][3]
            final_pose.position.y = pose[1][3]
            final_pose.position.z = pose[2][3]
            final_pose.orientation.x = rot_quat[0]
            final_pose.orientation.y = rot_quat[1]
            final_pose.orientation.z = rot_quat[2]
            final_pose.orientation.w = rot_quat[3]
            pose_results.append(final_pose)
            valid_class_names.append(name)
            class_confidences.append(confidence)
        
        response = GenericImgProcAnnotator.Result()
        response.pose_results = pose_results
        response.class_names = valid_class_names
        response.class_confidences = class_confidences

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Execution time:', elapsed_time, 'seconds')
        goal_handle.succeed()
        return response

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ycbv', help='name of the dataset')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    rclpy.init()
    opt = parse_opt()
    renderer_request_queue = queue.Queue()
    renderer_result_queue = queue.Queue()

    node = GDRN_ROS(renderer_request_queue, renderer_result_queue, **vars(opt))
    node.set_parameters([
    rclpy.parameter.Parameter(
        "use_sim_time",
        rclpy.Parameter.Type.BOOL,
        True
    )])
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    executor.spin()   # run in main thread
    
    # Load camera intrinsics from file
    #intrinsics = np.asarray(rospy.get_param('/pose_estimator/intrinsics'))
    #renderer = RendererROS((64, 64), intrinsics, model_paths=None, scale_to_meter=1.0, gpu_id=None)

    #while not rospy.is_shutdown():
    #    if not renderer_request_queue.empty():
    #        request = renderer_request_queue.get(block=True, timeout=0.2)

    #        K_crop  = request[0]
    #        model = request[1]
    #        pose_est = request[2]
    #        renderer.clear() 
    #        renderer.set_cam(K_crop)
    #        renderer.draw_model(model,pose_est)
    #        _, ren_dp = renderer.finish()
    #        renderer_result_queue.put(ren_dp)
    #    else:
    #        rospy.sleep(0.1)

    
