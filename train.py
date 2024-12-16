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

t = time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
# 设置超参数
GPU_name = torch.cuda.get_device_name(0)
model_name = 'ultralytics/cfg/models/military/yolo11.yaml'
epochs = 150
if '3060' in GPU_name:
    batch = 16
elif '3080' in GPU_name:
    batch = 32
elif '4090' in GPU_name:
    batch = 32
elif '3090' in GPU_name:
    batch = 32
name = f'{model_name[model_name.rfind("/")+1:].split(".")[0]}-SGD-{epochs}-{batch}-{time}'

model = YOLO(model=model_name)

model.train(
    model=model_name,
    epochs=epochs,
    batch=batch,
    name=name,
)