import torch
import cifar10

model_dict = {
    # 3 layer cifar resnets
    "resnet8": cifar10.resnet8,  # params: 89322  Cifar10: 0.89590
    "resnet14": cifar10.resnet14,  # params: 186538 Cifar10: 0.92180
    "resnet20": cifar10.resnet20,  # params: 283754 Cifar10: 0.93020
    "resnet26": cifar10.resnet26,  # params: 380970 Cifar10: 0.92180
    "resnet32": cifar10.resnet32,  # params: 478186 Cifar10: 0.93690
    "resnet44": cifar10.resnet44,  # params: 672618 Cifar10: 0.94400
    "resnet56": cifar10.resnet56,  # params: 867050 Cifar10: 0.94510
    # 4 layer cifar resnets
    "resnet10": cifar10.resnet10,  # params: 4903242 Cifar10: 0.94300
    "resnet18": cifar10.resnet18,  # params: 11173962 Cifar10: 0.95260
    "resnet34": cifar10.resnet34,  # params: 21282122
    "resnet50": cifar10.resnet50,  # params: 23520842
    "resnet101": cifar10.resnet101,  # params: 42512970
    "resnet152": cifar10.resnet152,  # params: 58156618
    # PreactResnt
    "PreActResNet18":cifar10.PreActResNet18,
    "PreActResNet34":cifar10.PreActResNet34,
}


def create_model(name, num_classes, device):
    model_cls = model_dict[name]
    print(f"Building model {name}...", end='')
    model = model_cls(num_classes=num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    layers = len(list(model.modules()))
    print(f" total parameters: {total_params}, layers {layers}")
    # always use dataparallel for now
    model = torch.nn.DataParallel(model)
    device_count = torch.cuda.device_count()
    print(f"Using {device_count} GPU(s).")
    # copy to cuda if activated
    model = model.to(device)
    return model

if __name__ == "__main__":
    for model in model_dict.keys():
        create_model(model, 10, "cpu")
