from ultralytics import YOLO

# 加载你刚才练好的模型
model = YOLO("runs/detect/tactix_train/football_v12/weights/best.pt")

# 打印它学到的类别
print(model.names)