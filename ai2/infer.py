#!/usr/bin/env python3
import cv2
import numpy as np
import hailo_platform as hpf

class HailoYoloDetector:
    def __init__(self, model_path, classes, conf_threshold=0.01, iou_threshold=0.45):
        self.classes        = classes
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold

        # 1) HEF 로드
        self.hef = hpf.HEF(model_path)

        # 2) VDevice 생성 및 configure
        self.device = hpf.VDevice()
        cfg = hpf.ConfigureParams.create_from_hef(
            self.hef, interface=hpf.HailoStreamInterface.PCIe
        )
        self.network_group  = self.device.configure(self.hef, cfg)[0]
        self.network_params = self.network_group.create_params()

        # 3) I/O stream info
        self.input_info  = self.hef.get_input_vstream_infos()[0]
        self.output_info = self.hef.get_output_vstream_infos()[0]

        # 4) InputVStreamParams: UINT8 quantized
        self.input_params = hpf.InputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=True,
            format_type=hpf.FormatType.UINT8
        )
        # 모델 스크립트에서 [0..255] 그대로 normalization 했으므로
        self.scale      = 1.0
        self.zero_point = 0

        # 5) OutputVStreamParams: float32
        self.output_params = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=False,
            format_type=hpf.FormatType.FLOAT32
        )

        # 6) 입력 해상도
        shape = self.input_info.shape
        if len(shape)==4:
            _,_,h,w = shape
        else:
            h,w,_ = shape
        self.input_height, self.input_width = h, w

    def preprocess(self, frame):
        # letterbox resize
        ih, iw = frame.shape[:2]
        scale = min(self.input_width/iw, self.input_height/ih)
        nw, nh = int(iw*scale), int(ih*scale)
        resized = cv2.resize(frame, (nw, nh))
        # padding
        dw, dh = self.input_width - nw, self.input_height - nh
        top, left = dh//2, dw//2
        padded = cv2.copyMakeBorder(resized, top, dh-top, left, dw-left,
                                    cv2.BORDER_CONSTANT, value=(0,0,0))
        # BGR→RGB, NHWC, uint8
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb[np.newaxis, ...].astype(np.uint8)
        return tensor, scale, left, top, iw, ih

    def infer(self, tensor):
        with self.network_group.activate(self.network_params):
            with hpf.InferVStreams(self.network_group,
                                   self.input_params,
                                   self.output_params) as pipeline:
                results = pipeline.infer({self.input_info.name: tensor})
        return results[self.output_info.name]

    def postprocess(self, outputs, scale, pad_left, pad_top, ow, oh):
        # 1) flat outputs
        raw = outputs if isinstance(outputs,(list,tuple)) else [outputs]
        flat = []
        for o in raw:
            if isinstance(o,(list,tuple)): flat.extend(o)
            else: flat.append(o)

        # 2) normalized → padded → unpad → 원본 coords
        all_boxes, all_scores, all_cls = [], [], []
        for cls_idx, arr in enumerate(flat):
            arr = np.array(arr, dtype=np.float32)
            if arr.ndim!=2 or arr.shape[1]!=5: 
                continue
            for x1n,y1n,x2n,y2n,conf in arr:
                # filter by conf
                if conf < self.conf_threshold: 
                    continue
                # normalized → pixel in padded frame
                px1,py1 = x1n*self.input_width,  y1n*self.input_height
                px2,py2 = x2n*self.input_width,  y2n*self.input_height
                # remove padding, back to original scale
                ux1 = (px1 - pad_left)/scale
                uy1 = (py1 - pad_top )/scale
                ux2 = (px2 - pad_left)/scale
                uy2 = (py2 - pad_top )/scale
                # clamp to image
                if ux1<0 or uy1<0 or ux2>ow or uy2>oh: 
                    continue
                all_boxes.append([ux1, uy1, ux2, uy2])
                all_scores.append(float(conf))
                all_cls.append(cls_idx)

        # **디버깅: 후보 박스/스코어 출력**
        print(">>> Candidates before NMS:")
        for b,s,c in zip(all_boxes, all_scores, all_cls):
            print(f"  box={b}, score={s:.3f}, cls={self.classes[c]}")

        detections = []
        if all_boxes:
            # convert to [x,y,w,h]
            rects = []
            for (x1,y1,x2,y2) in all_boxes:
                w = x2 - x1
                h = y2 - y1
                rects.append([x1, y1, w, h])
            # **디버그: rects 리스트 출력**
            print(">>> rects for NMS:", rects)
            # OpenCV NMS
            idxs = cv2.dnn.NMSBoxes(rects, all_scores,
                                    self.conf_threshold,
                                    self.iou_threshold)
            if len(idxs)>0:
                for i in idxs.flatten():
                    x1,y1,x2,y2 = all_boxes[i]
                    detections.append((x1,y1,x2,y2,
                                       all_scores[i],
                                       all_cls[i]))
        return detections

    def detect(self, frame):
        tensor, scale, pad_x, pad_y, ow, oh = self.preprocess(frame)
        raw = self.infer(tensor)
        return self.postprocess(raw, scale, pad_x, pad_y, ow, oh)

    def close(self):
        try: self.device.close()
        except: pass


def main():
    MODEL_PATH = "/home/ssafy/project/RPI-tests/ai/models/yolov8n_level0.hef"
    CLASSES    = ["Bee","Wasp"]
    IMAGE_PATH = "/home/ssafy/project/RPI-tests/ai2/example.jpg"

    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print("Cannot load image:", IMAGE_PATH)
        return

    detector = HailoYoloDetector(
        MODEL_PATH,
        CLASSES,
        conf_threshold=0.25,   # 낮춰서 raw 후보 확인
        iou_threshold=0.45
    )

    dets = detector.detect(frame)
    print(f"--- Final detections (after NMS): {len(dets)} ---")
    for x1,y1,x2,y2,conf,cls in dets:
        label = CLASSES[cls]
        color = (0,0,255) if label=="Wasp" else (255,0,0)
        xi1, yi1, xi2, yi2 = map(int,(x1,y1,x2,y2))
        cv2.rectangle(frame,(xi1,yi1),(xi2,yi2), color, 2)
        cv2.putText(frame, f"{label} {conf*100:.1f}%",
                    (xi1, yi1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out = "/home/ssafy/result_debug.jpg"
    cv2.imwrite(out, frame)
    print("Saved:", out)
    detector.close()


if __name__=="__main__":
    main()
