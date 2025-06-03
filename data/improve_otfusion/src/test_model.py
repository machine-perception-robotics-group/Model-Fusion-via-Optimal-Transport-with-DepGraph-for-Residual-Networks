import torch
import torch.nn as nn


def test(model, test_loader, device=None, verbose=False):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    loss_function = nn.CrossEntropyLoss(reduction="sum")

    correct_1 = 0.0
    correct_5 = 0.0
    loss_total = 0.0

    with torch.no_grad():
        for image, label in test_loader:

            image = image.to(device)
            label = label.to(device)

            output = model(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            loss = loss_function(output, label)
            loss_total += loss.item()

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            # compute top 5
            correct_5 += correct[:, :5].sum()
            # compute top1
            correct_1 += correct[:, :1].sum()

    if verbose:
        if device == "cuda":
            print("GPU INFO.....")
            print(torch.cuda.memory_summary(), end="")

        print()
        print("Top 1 err: ", 1 - correct_1 / len(test_loader.dataset))
        print("Top 5 err: ", 1 - correct_5 / len(test_loader.dataset))
        print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))

    top1_acc = correct_1 / len(test_loader.dataset)
    top5_acc = correct_5 / len(test_loader.dataset)
    test_loss = loss_total / len(test_loader.dataset)

    return top1_acc.item(), top5_acc.item(), test_loss
