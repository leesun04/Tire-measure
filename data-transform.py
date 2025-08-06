import json
import os

def convert_json_folder_to_yolo_keypoints(input_json_folder, output_txt_folder): #레이블미에서 나온 json 파일들을 YOLO keypoints 형식으로 변환하는 함수
    os.makedirs(output_txt_folder, exist_ok=True)

    json_files = [f for f in os.listdir(input_json_folder) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(input_json_folder, json_file)
        output_txt_path = os.path.join(output_txt_folder, os.path.splitext(json_file)[0] + '.txt')

        with open(json_path, 'r') as f:
            data = json.load(f)

        imageWidth = data.get('imageWidth')
        imageHeight = data.get('imageHeight')

        if imageWidth is None or imageHeight is None:
            print(f"[경고] {json_file}에 imageWidth 또는 imageHeight 정보가 없습니다. 건너뜁니다.")
            continue

        class_id = 0

        keypoints = []

        # 'tire-width' 2점 추출
        for shape in data['shapes']:
            if shape['label'] == 'tire-width':
                for pt in shape['points']:
                    x_rel = pt[0] / imageWidth
                    y_rel = pt[1] / imageHeight
                    visible = 2
                    keypoints.extend([x_rel, y_rel, visible])
                break

        # 'tire-hight' 2점 추출
        for shape in data['shapes']:
            if shape['label'] == 'tire-hight':
                for pt in shape['points']:
                    x_rel = pt[0] / imageWidth
                    y_rel = pt[1] / imageHeight
                    visible = 2
                    keypoints.extend([x_rel, y_rel, visible])
                break

        # bbox 추출
        bbox = None
        for shape in data['shapes']:
            if shape['label'] == 'tire':
                x1, y1 = shape['points'][0]
                x2, y2 = shape['points'][1]
                x_center = ((x1 + x2) / 2) / imageWidth
                y_center = ((y1 + y2) / 2) / imageHeight
                width = abs(x2 - x1) / imageWidth
                height = abs(y2 - y1) / imageHeight
                bbox = (x_center, y_center, width, height)
                break

        if bbox is None:
            print(f"[경고] {json_file}에 'tire' 라벨이 없습니다. 건너뜁니다.")
            continue

        line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} "
        line += " ".join(f"{kp:.6f}" if i % 3 != 2 else str(int(kp)) for i, kp in enumerate(keypoints))

        with open(output_txt_path, 'w') as f:
            f.write(line + "\n")

        print(f"변환 완료: {output_txt_path}")

# 사용 예
input_json_folder = "" # JSON 파일들이 있는 디렉토리
output_txt_folder = "" # 원하는 저장 폴더로 변경
convert_json_folder_to_yolo_keypoints(input_json_folder, output_txt_folder)
