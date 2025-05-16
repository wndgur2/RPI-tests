import cv2
import numpy as np
import hailo_platform as hpf


def letterbox(image, new_shape=(640, 640), color=(0, 0, 0), scaleup=True):
    shape = image.shape[:2]  # (h, w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # (w, h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    img = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, ratio, (dw, dh)


class HailoYoloDetector:
    def __init__(self, model_path, classes, conf_threshold=0.25, iou_threshold=0.45):
        self.classes = classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # HEF 모델 로딩
        self.hef = hpf.HEF(model_path)
        self.device = hpf.VDevice()

        # 인터페이스 자동 선택
        cfg_params = hpf.ConfigureParams.create_from_hef(
            self.hef,
            interface=hpf.HailoStreamInterface.PCIe
        )

        self.network_group = self.device.configure(self.hef, cfg_params)[0]
        self.network_params = self.network_group.create_params()

        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_info = self.hef.get_output_vstream_infos()[0]

        shape = self.input_info.shape  # [H, W, C]
        self.input_height, self.input_width = shape[0], shape[1]

        self.input_params = hpf.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=True
        )
        self.output_params = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=True
        )

    def preprocess(self, frame):
        """Letterbox 리사이즈 + NCHW 변환 + float32"""
        padded, (scale_x, scale_y), (dw, dh) = letterbox(
            frame, new_shape=(self.input_width, self.input_height)
        )


        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        orig_h, orig_w = frame.shape[:2]
        return tensor, scale_x, dw, dh, orig_w, orig_h

    def infer(self, tensor: np.ndarray):
        with self.network_group.activate(self.network_params):
            with hpf.InferVStreams(self.network_group, self.input_params, self.output_params) as pipeline:
                input_data = {self.input_info.name: tensor}
                results = pipeline.infer(input_data)
        return results[self.output_info.name]

    def postprocess(self, outputs, scale, pad_left, pad_top, orig_w, orig_h):
        detections = []

        if not isinstance(outputs, (list, tuple)) or not isinstance(outputs[0], (list, tuple)):
            print("[ERROR] Unexpected output format")
            return []

        for cls_id, class_out in enumerate(outputs[0]):
            arr = np.array(class_out, dtype=np.float32)

            if arr.ndim != 2 or arr.shape[1] != 5:
                print(f"[WARN] Unexpected shape: {arr.shape}")
                continue

            for x1_n, y1_n, x2_n, y2_n, score in arr:
                if score < self.conf_threshold:
                    continue
                print(x1_n, y1_n, x2_n, y2_n, score)
                px1 = x1_n * self.input_width
                py1 = y1_n * self.input_height
                px2 = x2_n * self.input_width
                py2 = y2_n * self.input_height

                ux1 = (px1 - pad_left) / scale
                uy1 = (py1 - pad_top) / scale
                ux2 = (px2 - pad_left) / scale
                uy2 = (py2 - pad_top) / scale

                if ux1 < 0 or uy1 < 0 or ux2 > orig_w or uy2 > orig_h:
                    continue

                detections.append((ux1, uy1, ux2, uy2, score, cls_id))

        print(f"[INFO] Detected {len(detections)} objects", flush=True)
        return detections

    def detect(self, frame):
        print("[INFO] Frame: Running detection...", flush=True)
        print("[DEBUG] Preprocessing frame...", flush=True)
        tensor, scale, pad_left, pad_top, w, h = self.preprocess(frame)
        print("[DEBUG] Running inference...", flush=True)
        raw = self.infer(tensor)
        print("[DEBUG] Postprocessing outputs...", flush=True)
        return self.postprocess(raw, scale, pad_left, pad_top, w, h)

    def close(self):
        try:
            self.device.close()
        except Exception:
            pass
