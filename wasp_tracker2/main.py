from camera.oak_camera import OakCamera
from infer.hailo_infer import HailoInfer
from tracker.tracker import WaspTracker
from utils.overlay import overlay_detections
from utils.class_names import CLASS_NAMES
import cv2
import time

def main():
    camera = OakCamera()
    infer = HailoInfer("models/yolov8n_coco.hef")
    tracker = WaspTracker()

    while True:
        frame, depth = camera.get_frames()
        print('camera.get_frames: ')
        print(frame.shape)

        ###################### 임시 사진 ############################
        # frame = cv2.imread("sample2.jpg")       
        ############################################################
        # TODO
        if frame is None or depth is None:
            continue
        
        detections = infer.run_inference(frame)
        tracked = tracker.update(detections)

        annotated = overlay_detections(frame.copy(), tracked, depth, camera.get_intrinsics(), CLASS_NAMES)
        cv2.imwrite("output.jpg", annotated)
        print("[INFO] Frame saved to output.jpg")
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()