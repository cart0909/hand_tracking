import argparse
import os.path as osp
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from pytorch.handsegnet import HandSegNet
from pytorch.segdataset import SegDataset
from torchvision import transforms

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

def train(model, dataloader, epoch, n_epochs, loss_function, optimizer, device):
    model.train()
    loop = tqdm(dataloader)
    for [img, mask] in loop:
        img = img.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        pred_mask = model(img)

        loss = loss_function(pred_mask, mask)
        
        loss.backward()
        optimizer.step()
        
        loop.set_description('Epoch {}/{}'.format(epoch + 1, n_epochs))
        loop.set_postfix(loss=loss.item())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--cpu', type=bool, default=False)
    parser.add_argument('--dataset_root', type=str, default='.')
    args = parser.parse_args()

    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
        ])

    train_dataset = SegDataset(args.dataset_root, output_size=256, transform=transform)
    train_dataloader = DataLoader(train_dataset, args.batch_size)

    test_dataset = SegDataset(args.dataset_root, is_train=False, output_size=256, transform=transform)
    test_dataloader = DataLoader(test_dataset, args.batch_size)

    model = HandSegNet().to(device)
    model.apply(weights_init)

    if osp.isfile('handsegnet.pth'):
        print('find the checkpoint model! loading...')
        model.load_state_dict(torch.load('handsegnet.pth'))

    optimizer = torch.optim.Adam(model.parameters())

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train(model, train_dataloader, epoch, args.epochs, loss_fn, optimizer, device)
        torch.save(model.state_dict(), 'handsegnet.pth')

if __name__ == '__main__':
    main()