from hailo_platform import HEF, VDevice, ConfigureParams, InferVStreams, \
    InputVStreamParams, OutputVStreamParams, FormatType, HailoStreamInterface

class HailoInfer:
    def __init__(self, hef_path):
        self.hef = HEF(hef_path)
        self.device = VDevice()
        self.network_group = self.device.configure(
            self.hef,
            ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        )[0]
        self.input_name = self.hef.get_input_vstream_infos()[0].name
        self.model_shape = self.hef.get_input_vstream_infos()[0].shape
        self.input_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        self.output_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )

    def run_inference(self, frame):
        import cv2
        import numpy as np

        h, w = self.model_shape[:2]
        resized = cv2.resize(frame, (w, h)).astype('float32') / 255.0
        input_tensor = {self.input_name: np.expand_dims(resized, axis=0)}

        with self.network_group.activate():
            with InferVStreams(self.network_group, self.input_params, self.output_params, tf_nms_format=True) as vstreams:
                output = vstreams.infer(input_tensor)

        return self.parse_output(output, frame.shape[:2])


    def parse_output(self, output, original_shape):
        results = []
        out = list(output.values())[0]
        H, W = original_shape
        print(f"[DEBUG] Raw output shape: {out.shape}")  # 로그 추가

        # 출력 내용 일부 확인
        for cls_id in range(out.shape[1]):
            for i in range(out.shape[3]):
                det = out[0, cls_id, :, i]  # shape (5,)
                conf = det[4]
                if conf > 0:
                    print(f"[DEBUG] CLASS {cls_id} | DET {i} | CONF = {conf:.4f} | BBOX = {det[:4]}")
                if conf > 0.1:
                    x1 = int(det[0] * W)
                    y1 = int(det[1] * H)
                    x2 = int(det[2] * W)
                    y2 = int(det[3] * H)
                    results.append((x1, y1, x2, y2, float(conf), cls_id))

        print(f"[INFO] Total detections over 0.1: {len(results)}")
        return results
