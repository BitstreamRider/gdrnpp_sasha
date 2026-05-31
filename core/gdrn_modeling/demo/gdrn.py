import json
import sys
import numpy as np
import cv2
import torch
import queue
import time

import threading

from predictor_gdrn import GdrnPredictor
from lib.render_vispy.renderer import RendererROS

stop_renderer = False

def main():
    input_path = sys.argv[1]

    with open(input_path, "r") as f:
        data = json.load(f)

    image_rgb = cv2.imread(data["image_path_rgb"])
    image_depth = np.load(data["image_path_depth"])
    image_depth = image_depth / 1000 # keeping it raw

    renderer_request_queue = queue.Queue()
    renderer_result_queue = queue.Queue()
    
    # start renderer thread
    worker_thread = threading.Thread(
        target=renderer_worker,
        args=(renderer_request_queue, renderer_result_queue, np.array(data["intrinsics"])),
        daemon=True
    )
    worker_thread.start()

    predictor = GdrnPredictor(
        config_file_path=data["config"],
        ckpt_file_path=data["weights"],
        camera_intrinsics=np.array(data["intrinsics"]),
        path_to_obj_models=data["models"],
        model_string=data["dataset"]
    )
    name = data["name"]
    ymin = data["ymin"]
    xmin = data["xmin"]
    ymax = data["ymax"]
    xmax = data["xmax"]
    score = data["score"]
    reflow = data["reflow"]
    plane_normal = data["plane_normal"]
    
    obj_id = -1
    for number in predictor.objs:
        if predictor.objs[number] == name:
            obj_id = int(number) 
            break
    assert obj_id > 0

    outputs = torch.tensor([float(ymin), float(xmin), float(ymax), float(xmax),  score, score, float(obj_id - 1)])
    outputs = list((outputs.unsqueeze(0)))
    
    data_dict = predictor.preprocessing(outputs=outputs, image=image_rgb, depth_img=image_depth)
    out_dict = predictor.inference(data_dict)
    poses = predictor.postprocessing(data_dict, out_dict, reflow=reflow, plane_normal=plane_normal, renderer_request_queue=renderer_request_queue, renderer_result_queue=renderer_result_queue)
    renderer_request_queue.put(None)
    stop_renderer = True
    if worker_thread is not None:
        worker_thread.join(timeout=1.0)
    
    obj_name = predictor.objs[int(obj_id)]
    # Convert to JSON serializable
    result = {
        "poses": {},
        "obj_names": obj_name
    }
    for k, v in poses.items():
        result["poses"][k] = v.tolist()

    print(json.dumps(result))


def renderer_worker(request_queue, result_queue, intrinsics):
    renderer = RendererROS(
        (64, 64),
        intrinsics,
        model_paths=None,
        scale_to_meter=1000,
        gpu_id=None
    )

    print("[Renderer] Worker started")

    while True:
        if stop_renderer:
            break
        try:
            request = request_queue.get(timeout=0.1)
            print("Queue size:", request_queue.qsize())
        except queue.Empty:
            time.sleep(0.01)
            continue

        K_crop, model, pose_est = request

        renderer.clear()
        renderer.set_cam(K_crop)
        renderer.draw_model(model, pose_est)

        _, ren_dp = renderer.finish()

        result_queue.put(ren_dp)
        

if __name__ == "__main__":
    main()
