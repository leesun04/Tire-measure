from ultralytics import YOLO

#model = YOLO("/mnt/nas4/lsj/Tire-test/runs/train_tire10/weights/best.pt")
model = YOLO("모델 yaml파일 경로")
print("모델 정보")
model.info()


model.train(
    data="데이터 yaml파일 경로",
    epochs = 200,
    imgsz = 640,
    batch = 5,
    project = "결과 프로젝트 경로",
    name="train_tire"
)