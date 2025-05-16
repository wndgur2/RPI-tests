import cv2
import numpy as np
import hailo_platform as hpf

class HailoYoloDetector:
    def __init__(self, model_path, classes, conf_threshold=0.25, iou_threshold=0.45):
        self.classes = classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.hef = hpf.HEF(model_path)
        self.device = hpf.VDevice()

        configure_params = hpf.ConfigureParams.create_from_hef(
            self.hef, interface=hpf.HailoStreamInterface.PCIe
        )
        self.network_group = self.device.configure(self.hef, configure_params)[0]
        self.network_params = self.network_group.create_params()

        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_infos = self.hef.get_output_vstream_infos()

        print("===== DEBUG HEF OUTPUT VSTREAM INFOS =====")
        for info in self.output_infos:
            print(f"  name={info.name}, shape={info.shape}")
        print("==========================================")

        self.input_height, self.input_width = self.input_info.shape[0], self.input_info.shape[1]

        self.input_params = hpf.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=True, format_type=hpf.FormatType.UINT8
        )
        self.output_params = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )

    def preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.input_width / w, self.input_height / h)
        nw, nh = int(w * scale), int(h * scale)
        pad_w, pad_h = self.input_width - nw, self.input_height - nh
        left, top = pad_w // 2, pad_h // 2

        resized = cv2.resize(frame, (nw, nh))
        padded = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                                    borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # NCHW
        return img.astype(np.uint8)

    def infer(self, input_tensor):
        with self.network_group.activate(self.network_params):
            with hpf.InferVStreams(self.network_group, self.input_params, self.output_params) as pipeline:
                results = pipeline.infer({self.input_info.name: input_tensor})
        return [results[o.name] for o in self.output_infos]

    def postprocess(self, outputs):
        detections = []

        # 안전하게 여러 단계의 리스트 언랩 처리
        raw_output = outputs[0]
        while isinstance(raw_output, list):
            if len(raw_output) == 0:
                print("[ERROR] Empty list in output")
                return detections
            raw_output = raw_output[0]

        if not isinstance(raw_output, np.ndarray):
            print(f"[ERROR] Final output is not ndarray: {type(raw_output)}")
            return detections

        print(f"[DEBUG] raw output shape: {raw_output.shape}")
        print(f"[DEBUG] output dtype: {raw_output.dtype}")

        if raw_output.ndim != 3 or raw_output.shape[0] != 2 or raw_output.shape[2] != 5:
            print(f"[WARN] Unexpected shape: {raw_output.shape}")
            return detections

        boxes_scores = raw_output[0]  # shape (N, 5)
        class_ids_raw = raw_output[1]  # shape (N, 5)

        for i in range(boxes_scores.shape[0]):
            x1, y1, x2, y2, score = boxes_scores[i]
            if score < self.conf_threshold:
                continue
            cls_id = int(class_ids_raw[i][0]) if i < class_ids_raw.shape[0] else 0
            detections.append((x1, y1, x2, y2, score, cls_id))

        return detections






    def detect(self, frame):
        input_tensor = self.preprocess(frame)
        raw_outputs = self.infer(input_tensor)
        return self.postprocess(raw_outputs)

    def close(self):
        try:
            self.device.close()
        except:
            pass
