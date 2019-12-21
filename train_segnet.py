import argparse
import os.path as osp
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from pytorch.handsegnet import HandSegNet
from pytorch.segdataset import SegDataset
from torchvision import transforms
import cv2
import numpy as np

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

def train(model, dataloader, epoch, n_epochs, loss_function, optimizer, device):
    model.train()
    loop = tqdm(dataloader)
    for img, mask in loop:
        img = img.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        pred_mask = model(img)

        loss = loss_function(pred_mask, mask)
        
        loss.backward()
        optimizer.step()
        
        loop.set_description('Epoch {}/{}'.format(epoch + 1, n_epochs))
        loop.set_postfix(loss=loss.item())

def eval(model, dataloader, loss_function, device):
    model.eval()
    loop = tqdm(dataloader)
    total_loss = 0
    save_first = True
    with torch.no_grad():
        for img, mask in loop:
            img = img.to(device)
            mask = mask.to(device)
            out = model(img)

            total_loss += loss_function(out, mask).item()
            pred_mask = out.argmax(dim=1)

            # save
            if save_first:
                unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                img = unorm(img)
                img = img.view(3, 256, 256)
                img = img.permute(1, 2, 0)
                pred_mask = pred_mask.type(torch.float32)
                pred_mask = pred_mask.view(1, 256, 256)
                pred_mask = pred_mask.permute(1, 2, 0)
                save_first = False

                img = np.uint8(img.cpu().numpy() * 255)
                pred_mask = np.uint8(pred_mask.cpu().numpy() * 255)

                cv2.imwrite('img.png', img)
                cv2.imwrite('mask.png', pred_mask)

    total_loss /= len(dataloader)
    print('Testset ave loss: {}'.format(total_loss))
    return total_loss

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
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)

    test_dataset = SegDataset(args.dataset_root, is_train=False, output_size=256, transform=transform)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=True)

    model = HandSegNet().to(device)
    model.apply(weights_init)

    if osp.isfile('handsegnet.pth'):
        print('find the checkpoint model! loading...')
        model.load_state_dict(torch.load('handsegnet.pth'))

    optimizer = torch.optim.Adam(model.parameters())

    loss_fn = torch.nn.CrossEntropyLoss()

    best_loss = -1
    for epoch in range(args.epochs):
        train(model, train_dataloader, epoch, args.epochs, loss_fn, optimizer, device)
        eval_loss = eval(model, test_dataloader, loss_fn, device)
        
        if best_loss > eval_loss or best_loss == -1:
            print('best loss {} bigger than eval loss {}. save the model'.format(best_loss, eval_loss))
            best_loss = eval_loss
            torch.save(model.state_dict(), 'handsegnet.pth')

if __name__ == '__main__':
    main()