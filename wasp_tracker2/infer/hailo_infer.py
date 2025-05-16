from hailo_platform import HEF, VDevice, ConfigureParams, InferVStreams, \
    InputVStreamParams, OutputVStreamParams, FormatType, HailoStreamInterface
import numpy as np
import cv2

class HailoInfer:
    def __init__(self, hef_path):
        self.hef = HEF(hef_path)
        self.device = VDevice()

        # 🔽 network_group 설정
        self.network_group = self.device.configure(
            self.hef,
            ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        )[0]

        # 🔽 input 정보 추출
        input_info = self.hef.get_input_vstream_infos()[0]
        print("[INFO] Input shape:", input_info.shape)
        print("[INFO] Input format:", input_info.format)  # order, type 등
        self.input_name = input_info.name
        self.model_shape = input_info.shape

        # 🔽 포맷 확인 예시 (선택)
        # if input_info.format.order == dai.HailoFormatOrder.NHWC:
        #     print("NHWC 형식")

        self.input_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        self.output_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )


    def run_inference(self, frame):
        h, w = self.model_shape[:2]

        # debugging: 입력 이미지 확인
        print("[DEBUG] Original frame shape:", frame.shape)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (w, h)).astype('float32') / 255.0
        print("[DEBUG] After resize and normalize:", resized.shape, resized.dtype)
        resized_chw = resized.transpose(2, 0, 1)  # (HWC) -> (CHW)
        print("[DEBUG] After transpose to CHW:", resized_chw.shape)
        input_tensor = {self.input_name: np.expand_dims(resized_chw, axis=0).copy()}
        print("[DEBUG] Final input_tensor shape:", input_tensor[self.input_name].shape)


        with self.network_group.activate():
            with InferVStreams(self.network_group, self.input_params, self.output_params, tf_nms_format=True) as vstreams:
                output = vstreams.infer(input_tensor)

        return self.parse_output(output, frame.shape[:2])


    def parse_output(self, output, original_shape):
        output_array = list(output.values())[0]  # shape: (1, 80, 5, 100)
        print(f"[DEBUG] Output shape: {output_array.shape}")

        H, W = original_shape
        results = []

        for cls_id in range(output_array.shape[1]):  # 80
            for i in range(output_array.shape[3]):  # 100
                det = output_array[0, cls_id, :, i]  # (5,)
                conf = det[4]
                if conf > 0.3:
                    x1 = int(det[0] * W)
                    y1 = int(det[1] * H)
                    x2 = int(det[2] * W)
                    y2 = int(det[3] * H)
                    results.append((x1, y1, x2, y2, float(conf), cls_id))

        print(f"[INFO] Total detections: {len(results)}")
        return results



