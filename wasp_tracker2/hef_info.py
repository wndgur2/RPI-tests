from hailo_platform import HEF

hef = HEF("models/yolov8n2.hef")

# 입력 스트림 정보
input_infos = hef.get_input_vstream_infos()
for info in input_infos:
    print(f"[INPUT] name: {info.name}, shape: {info.shape}, format: {info.format}")

# 출력 스트림 정보
output_infos = hef.get_output_vstream_infos()
for info in output_infos:
    print(f"[OUTPUT] name: {info.name}, shape: {info.shape}, format: {info.format}")

# 네트워크 이름
print(f"Network name: {hef.get_network_group_names()[0]}")
