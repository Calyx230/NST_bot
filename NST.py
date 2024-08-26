from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import copy


class Normalization(nn.Module):
    def __init__(self, mean, std, device):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean, device=device).view(-1, 1, 1)
        self.std = torch.tensor(std, device=device).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = 0

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class LaplacianLoss(nn.Module):

    def __init__(self, target, device):
        super(LaplacianLoss, self).__init__()
        laplacian = torch.Tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        self.conv = nn.Conv2d(3, 1, kernel_size=laplacian.size(), padding=1)
        for param in self.conv.parameters():
            param.requires_grad = False
        for i in range(3):
            self.conv.weight[0, i] = laplacian
        self.conv.to(device)
        self.loss = 0
        self.target = self.conv(target)

    def forward(self, x):
        inp = self.conv(x)
        self.loss = F.mse_loss(self.target, inp)
        return x


class StyleLoss(nn.Module):

    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target= self.gram_matrix(target).detach()
        self.loss = 0

    @staticmethod
    def gram_matrix(self, inp):
        # b for batch size, c for number of channels, h for height and w for width
        b, c, h, w = inp.size()
        feat = inp.view(b*h, w*c)
        den = b*c*h*w
        return torch.mm(feat, feat.t())/den

    def forward(self, x):
        inp = self.gram_matrix(x)
        self.loss = F.mse_loss(inp, self.target)
        return x


class NST:
    def __init__(self, content_layers, style_layers, laplacian_layers):

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.laplacian_layers = laplacian_layers
        self.losses = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_image(self, path):
        image = Image.open(path)
        prepare = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        image = prepare(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def build_model(self, cnn, content_img, style_img):
        cnn = copy.deepcopy(cnn)
        styleblocks, contentblocks, lap_blocks = [], [], []
        conv_count = 1
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalization = Normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], self.device)
        laplace = LaplacianLoss(content_img, self.device)
        lap_blocks = [laplace]
        model = nn.Sequential(normalization, laplace)
        for i, layer in enumerate(cnn.features):
            if isinstance(layer, nn.ReLU):
                model.add_module(f'relu_{i}', layer)
                if i in self.content_layers:
                    target = model(content_img)
                    new_block = ContentLoss(target)
                    model.add_module(f'content_loss_{i}', new_block)
                    contentblocks.append(new_block)
                if i in self.style_layers:
                    target = model(style_img)
                    new_block = StyleLoss(target)
                    model.add_module(f'style_loss_{i}', new_block)
                    styleblocks.append(new_block)
                conv_count += 1
            elif isinstance(layer, nn.Conv2d):
                model.add_module(f'conv_{i}', layer)
            elif isinstance(layer, nn.MaxPool2d):
                model.add_module(f'pooling_{i}', layer)  # nn.AvgPool2d(2, 2))
            elif isinstance(layer, nn.BatchNorm2d):
                model.add_module(f'bn_{i}', layer)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], StyleLoss) or isinstance(model[i], ContentLoss) or isinstance(model[i],
                                                                                                  LaplacianLoss):
                break
        model = model[:(i + 1)]
        print(model)

        return model, styleblocks, contentblocks, lap_blocks

    def train(self, content, style, n_epochs=1000, alpha=500000, beta=1, gamma=1):
        content_target = self.load_image(content)
        style_target = self.load_image(style)
        cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).to(self.device).eval()
        for param in cnn.parameters():
            param.requires_grad = False
        model, styleblocks, contentblocks, lap_blocks = self.build_model(cnn, content_target, style_target)

        inp = content_target.clone()
        opt = optim.LBFGS([inp.requires_grad_()])
        run = [0]
        while run[0] < n_epochs:
            def closure():
                opt.zero_grad()
                inp.data.clamp_(0, 1)
                model(inp)
                style_loss = sum(block.loss for block in styleblocks)
                content_loss = sum(block.loss for block in contentblocks)
                lap_loss = sum(block.loss for block in lap_blocks)
                style_loss *= alpha
                content_loss *= beta
                lap_loss *= gamma
                loss = style_loss + content_loss + lap_loss
                loss.backward()
                run[0] += 1
                # if run[0] % 50 == 0:
                  # print(f'run {run}')
                  # print('Style loss {:4f}, Content loss {:4f}, Laplacian loss {:4f}'.format(style_loss.item(), content_loss.item(), lap_loss.item()))
                  # print()
                # self.losses.append(loss)
                return style_loss + content_loss + lap_loss
            opt.step(closure)

        inp = inp.cpu().detach()
        inp.data.clamp_(0, 1)
        self.save_picture(inp)

    def save_picture(self, tensor):
        unloader = transforms.ToPILImage()
        image = unloader(tensor.clone().squeeze(0))
        image.save("result.jpg")
