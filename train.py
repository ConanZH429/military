import comet_ml

from pathlib import Path
from ultralytics import YOLO
import datetime
import torch
import os
import yaml

# expoet COMET_MODE=offline
# export COMET_API_KEY=agcu7oeqU395peWf6NCNqnTa7

data_path = Path(__file__).parent.parent.absolute() / Path('military-data')

# 将data.yaml中的path路径改为绝对路径
data_yaml_path = './ultralytics/cfg/datasets/military.yaml'
with open(data_yaml_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    # data 中的 path 改为 绝对路径
    cfg['path'] = str(data_path)
    with open(data_yaml_path, 'w') as f:
        yaml.dump(cfg, f)

# 将当前文件夹下的train.txt, val.txt, test.txt拷贝到data_path下,若已存在则覆盖
os.system(f'cp ./train.txt {data_path}/train.txt')
os.system(f'cp ./val.txt {data_path}/val.txt')
os.system(f'cp ./test.txt {data_path}/test.txt')
os.environ["COMET_API_KEY"] = "agcu7oeqU395peWf6NCNqnTa7"
# 设置全局的git信息
os.system("git config --global user.email '863101876@qq.com'")
os.system("git config --global user.name 'ConanZH'")

t = time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
# 设置超参数
GPU_name = torch.cuda.get_device_name(0)
if '3060' in GPU_name:
    batch = 16
elif '3080' in GPU_name:
    batch = 32
elif '4090' in GPU_name:
    batch = 64
elif '3090' in GPU_name:
    batch = 64
    
model_name = 'ultralytics/cfg/models/military/yolo11-p2.yaml'
epochs = 5
os.environ["COMET_MODE"] = "offline"
name = f'{model_name[model_name.rfind("/")+1:].split(".")[0]}-SGD-{epochs}-{batch}-{time}'

model = YOLO(model=model_name)

model.train(
    model=model_name,
    epochs=epochs,
    batch=batch,
    name=name,
)

try:
    # val
    experiment = comet_ml.start(experiment_key=model.trainer.comet_key)
    label = {0: "tank", 1: "armored", 2: "truck", 3: "light"}

    best_path = f"./military/{name}/weights/best.pt"
    model = YOLO(best_path)
    test_name = "test-" + name
    metrics = model.val(split="test", name=test_name)
    metrics_dict = metrics.results_dict

    # 指标
    print("Uploading metrics...")
    metrics_dict["test-metrics/precision(B)"] = metrics_dict["metrics/precision(B)"]
    del metrics_dict["metrics/precision(B)"]
    metrics_dict["test-metrics/recall(B)"] = metrics_dict["metrics/recall(B)"]
    del metrics_dict["metrics/recall(B)"]
    metrics_dict["test-metrics/mAP50(B)"] = metrics_dict["metrics/mAP50(B)"]
    del metrics_dict["metrics/mAP50(B)"]
    metrics_dict["test-metrics/mAP50-95(B)"] = metrics_dict["metrics/mAP50-95(B)"]
    del metrics_dict["metrics/mAP50-95(B)"]
    metrics_dict["test-metrics/mAP75(B)"] = metrics.box.map75
    for i in range(4):
        metrics_dict[f"test-metrics/precision_{label[i]}(B)"] = metrics.box.p[i]
        metrics_dict[f"test-metrics/recall_{label[i]}(B)"] = metrics.box.r[i]
        metrics_dict[f"test-metrics/ap50_{label[i]}(B)"] = metrics.box.ap50[i]
        metrics_dict[f"test-metrics/ap50-95_{label[i]}(B)"] = metrics.box.maps[i]
        metrics_dict[f"test-metrics/f1_{label[i]}(B)"] = metrics.box.f1[i]

    metrics_dict[f"test-metrics/f1(B)"] = metrics.box.f1.mean()
    experiment.log_metrics(metrics_dict)
    print("Uploading metrics done.")
    # 混淆矩阵
    print("Uploading confusion matrix...")
    experiment.log_confusion_matrix(
        matrix=metrics.confusion_matrix.matrix.astype(int).tolist(),
        labels=["tank", "armored", "truck", "light", "background"],
        file_name="test-confusion-matrix.json"
    )
    print("Uploading confusion matrix done.")
    # 图像
    val_path = Path(f"./military/{test_name}")
    # 将val_path下的图片上传到comet
    for img in val_path.iterdir():
        img_name = img.name
        if "val" in img_name:
            img_name = img_name.replace("val_", "test")
        else:
            img_name = "test_" + img_name
        experiment.log_image(str(img), name=img_name)
except Exception as e:
    print(e)
finally:
    os.system("/usr/bin/shutdown")