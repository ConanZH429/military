from timm import create_model

model = create_model("mobilenetv3_large_100", pretrained=False)
print(model)