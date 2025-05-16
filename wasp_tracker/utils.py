import cv2

def overlay_detections(frame, tracked, depth, intrinsics, class_names):
    fx, fy, cx, cy = intrinsics
    for i, box in enumerate(tracked.xyxy):
        x1, y1, x2, y2 = map(int, box)
        tid = int(tracked.tracker_id[i])
        cls_id = int(tracked.class_id[i])
        label = class_names[cls_id]
        cx_pixel = min((x1 + x2) // 2, depth.shape[1] - 1)
        cy_pixel = min((y1 + y2) // 2, depth.shape[0] - 1)
        Z = float(depth[cy_pixel, cx_pixel])
        if Z > 0:
            X = (cx_pixel - cx) * Z / fx
            Y = (cy_pixel - cy) * Z / fy
            text = f"{label}#{tid}: ({X/1000:.2f}, {Y/1000:.2f}, {Z/1000:.2f})m"
            cv2.putText(frame, text, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return frame
