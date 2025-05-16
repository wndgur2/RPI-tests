import cv2
import numpy as np
import hailo_platform as hpf  # InferVDevice 대신 hailo_platform 전체를 import

class HailoYoloDetector:
    def __init__(self, model_path, classes, conf_threshold=0.25, iou_threshold=0.45):
        self.classes = classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 1) HEF 파일 로드
        self.hef = hpf.HEF(model_path)

        # 2) 가상 디바이스 생성
        self.device = hpf.VDevice()

        # 3) ConfigureParams 생성 (PCIe 인터페이스 사용 예시)
        cfg_params = hpf.ConfigureParams.create_from_hef(
            self.hef,
            interface=hpf.HailoStreamInterface.PCIe
        )

        # 4) 네트워크 그룹 생성 (리스트 중 첫 번째)
        self.network_group = self.device.configure(self.hef, cfg_params)[0]
        self.network_params = self.network_group.create_params()

        # 5) VStream 입/출력 정보
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_info = self.hef.get_output_vstream_infos()[0]

        print("===== DEBUG HEF INPUT VSTREAM INFOS =====")
        for _ in range(10):
            print("!!!!!!!!!!!")
        for info in self.hef.get_input_vstream_infos():
            print(f"  name={info.name}, shape={info.shape}")
        print("========================================")


        # 모델 입력 해상도 shape을 보고 H, W를 뽑아옵니다
        shape = self.input_info.shape
        h, w, _ = shape
        self.input_height, self.input_width = h, w

        # 6) VStream 파라미터 생성 (Float32 예시)
        self.input_params = hpf.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=True
        )
        self.output_params = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=True
        )

        # 모델 입력 크기 (예: [N, C, H, W])
        self.input_shape = tuple(self.input_info.shape)
        print("@@@@@")
        print(self.input_shape)
        # (후처리 시 필요에 따라 입력/출력 모양 활용)


        

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        """
        tensor: self.input_shape 에 맞춘 numpy 배열 (e.g. [1,3,640,640], float32)
        반환: output tensor (e.g. [1, ...], float32)
        """
        print(tensor.shape)
        # 네트워크 활성화 및 추론 스트림 생성
        with self.network_group.activate(self.network_params):
            with hpf.InferVStreams(
                self.network_group,
                self.input_params,
                self.output_params
            ) as infer_pipeline:
                # dict 형식으로 vstream name 에 매핑
                input_data = {self.input_info.name: tensor}
                results = infer_pipeline.infer(input_data)
        # 결과 리턴
        return results[self.output_info.name]

    def preprocess(self, frame):
        """프레임을 모델 입력 크기로 레터박스(letterbox)하고 배치 텐서로 변환."""
        img_h, img_w = frame.shape[0:2]
        scale = min(self.input_width / img_w, self.input_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 상하좌우 패딩
        pad_w = self.input_width - new_w
        pad_h = self.input_height - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        padded_img = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        # BGR→RGB, NCHW, float32
        rgb_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        input_tensor = np.transpose(rgb_img, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        # print(f"[DEBUG] Preprocessed tensor.shape={input_tensor.shape}, dtype={input_tensor.dtype}, nbytes={input_tensor.nbytes}")

        return input_tensor, scale, pad_left, pad_top, img_w, img_h


    def postprocess(self, outputs, scale, pad_left, pad_top, original_width, original_height):
        """모델 출력을 바운딩 박스, 클래스, 점수 리스트로 디코딩하고 NMS 적용."""
        # 1) flat outputs
        raw_list = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        flat = []
        for o in raw_list:
            if isinstance(o, (list,tuple)):
                flat.extend(o)
            else:
                flat.append(o)
        outputs_list = flat

        # 2) normalized → padded coords
        bboxes, scores, class_ids = [], [], []
        for cls_idx, out in enumerate(outputs_list):
            arr = np.array(out, dtype=np.float32)
            if arr.ndim!=2 or arr.shape[1]!=5:
                continue
            for x1_n, y1_n, x2_n, y2_n, score in arr:
                if score < self.conf_threshold:
                    continue
                # normalized → padded-input 픽셀
                px1 = x1_n * self.input_width
                py1 = y1_n * self.input_height
                px2 = x2_n * self.input_width
                py2 = y2_n * self.input_height

                # 3) letterbox 제거 & 원본 좌표로 변환
                ux1 = (px1 - pad_left) / scale
                uy1 = (py1 - pad_top ) / scale
                ux2 = (px2 - pad_left) / scale
                uy2 = (py2 - pad_top ) / scale

                if ux1 < 0 or uy1 < 0 or ux2 > original_width or uy2 > original_height:
                    continue
                # 정상 범위 안에 있으면 그대로 사용
                ox1, oy1, ox2, oy2 = ux1, uy1, ux2, uy2

                # --- 디버그 로그: 정확히 어떻게 변환되는지 한 번에 찍어 봅시다 ---
                print(f"[MAPPING] cls={self.classes[cls_idx]} | score={score:.2f}")
                print(f"  norm    -> ({x1_n:.3f},{y1_n:.3f},{x2_n:.3f},{y2_n:.3f})")
                print(f"  padded  -> ({px1:.1f},{py1:.1f},{px2:.1f},{py2:.1f})")
                print(f"  unpad   -> ({ux1:.1f},{uy1:.1f},{ux2:.1f},{uy2:.1f})")
                print(f"  original-> ({ox1:.1f},{oy1:.1f},{ox2:.1f},{oy2:.1f})")
                print("--------------------------------------------------------------")

                bboxes.append([ox1, oy1, ox2, oy2])
                scores.append(score)
                class_ids.append(cls_idx)
            # # (optional) 여전히 6+ 칼럼(클래스 id 포함) 형태가 섞여 있을 수도 있으니
            # if out_array.ndim == 2 and out_array.shape[1] >= 6:
            #     for det in out_array:
            #         x1, y1, x2, y2, score, cls_id = det[:6]
            #         if score < self.conf_threshold:
            #             continue
            #         bboxes.append([float(x1), float(y1), float(x2), float(y2)])
            #         scores.append(float(score))
            #         class_ids.append(int(cls_id))
        # ─────────────────────────────────────────────────────────────

        # raw reg_map & cls_map이 있으면 YOLO 스타일로 디코딩 (생략)
        # … (생략된 복잡한 YOLO 디코딩 로직) …

        # 클래스별 NMS 수행
        final_bboxes, final_scores, final_class_ids = [], [], []
        if len(bboxes) > 0:
            bboxes_arr = np.array(bboxes)
            scores_arr = np.array(scores)
            class_ids_arr = np.array(class_ids)
            for cls in np.unique(class_ids_arr):
                mask = class_ids_arr == cls
                boxes_cls = bboxes_arr[mask]
                scores_cls = scores_arr[mask]
                boxes_xywh = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes_cls]
                indices = cv2.dnn.NMSBoxes(
                    boxes_xywh, scores_cls.tolist(),
                    self.conf_threshold, self.iou_threshold
                )
                if len(indices) > 0:
                    for idx in indices.flatten():
                        final_bboxes.append(boxes_cls[idx])
                        final_scores.append(float(scores_cls[idx]))
                        final_class_ids.append(int(cls))

        # 레터박스→원본 크기로 역변환
        detections = []
        if len(bboxes)>0:
            boxes = np.array(bboxes)
            scores_arr = np.array(scores)
            cls_arr = np.array(class_ids)
            for cls in np.unique(cls_arr):
                mask = cls_arr == cls
                cls_boxes = boxes[mask]
                cls_scores = scores_arr[mask].tolist()
                # x,y,w,h for NMS
                xywh = [[x1,y1,x2-x1,y2-y1] for x1,y1,x2,y2 in cls_boxes]
                idxs = cv2.dnn.NMSBoxes(xywh, cls_scores,
                                        self.conf_threshold, self.iou_threshold)
                if len(idxs)>0:
                    for i in idxs.flatten():
                        x1,y1,x2,y2 = cls_boxes[i]
                        detections.append((x1,y1,x2,y2, float(cls_scores[i]), int(cls)))
        return detections

    def detect(self, frame):
        """전처리→추론→후처리를 한 번에 실행."""
        input_tensor, scale, pad_left, pad_top, w, h = self.preprocess(frame)
        raw = self.infer(input_tensor)
        return self.postprocess(raw, scale, pad_left, pad_top, w, h)

    def close(self):
        """Release the Hailo resources."""
        # VDevice를 닫아 줍니다.
        try:
            self.device.close()
        except:
            pass
