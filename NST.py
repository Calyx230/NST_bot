from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import copy


class Normalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class contentloss(nn.Module):

    def __init__(self, target, ):
        super().__init__()
        self.target = target.detach()
        self.loss = 0

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class styleloss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target= self.gram_matrix(target).detach()
        self.loss = 0
    def gram_matrix(self, input):
        b, c, h, w = input.size() #b for batch size, c for number of channels, h for height and w for width
        feat = input.view(b*h, w*c)
        den = b*c*h*w
        return torch.mm(feat, feat.t())/den
    def forward(self, x):
        inp = self.gram_matrix(x)
        self.loss = F.mse_loss(inp, self.target)
        return x

class NST:
    def __init__(self, content_layers, style_layers):
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.losses = []
    def load_image(self, path):
        image = Image.open(path)
        prepare = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        image = prepare(image).unsqueeze(0)
        return image
    
    def build_model(self, cnn, content_img, style_img):
        cnn = copy.deepcopy(cnn)
        styleblocks = []
        contentblocks = []
        model = nn.Sequential(Normalization()) 
        for i in range(41):
            if isinstance(cnn.features[i], nn.Conv2d): 
                model.add_module('conv_{}'.format(i), cnn.features[i])
                if i in self.content_layers:
                    target = model(content_img)
                    new_block = contentloss(target)
                    model.add_module('content_loss_{}'.format(i), new_block)
                    contentblocks.append(new_block)
                if i in self.style_layers:
                    target = model(style_img)
                    new_block = styleloss(target)
                    model.add_module('style_loss_{}'.format(i), new_block)
                    styleblocks.append(new_block) 
            elif isinstance(cnn.features[i], nn.ReLU):
                model.add_module('act_{}'.format(i), nn.ReLU(inplace=False))
            elif isinstance(cnn.features[i], nn.MaxPool2d):
                model.add_module('pooling_{}'.format(i), nn.AvgPool2d(2, 2))
            elif isinstance(cnn.features[i], nn.BatchNorm2d): 
                model.add_module('bn_{}'.format(i), cnn.features[i])
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))    
            
        return model, styleblocks, contentblocks

    def train(self, content, style, n_epochs=120, w1=100000, w2=1):
        content_target = self.load_image(content)
        inp = content_target.clone()
        style_target = self.load_image(style) 
        cnn = models.vgg16_bn(pretrained=True).eval()
        model, styleblocks, contentblocks = self.build_model(cnn, content_target, style_target)
        opt = optim.LBFGS([inp.requires_grad_()])
        run = [0]
        while run[0] < n_epochs:
            def closure():
                inp.data.clamp_(0, 1)
                opt.zero_grad()
                model(inp)
                style_loss = 0
                content_loss = 0
                for block in styleblocks:
                    style_loss += block.loss
                for bloc in contentblocks:
                    content_loss += bloc.loss
                style_loss*=w1
                content_loss*=w2
                loss = style_loss + content_loss
                loss.backward()
                run[0]+=1
                self.losses.append(loss)
                return style_loss + content_loss
            opt.step(closure)
        inp.data.clamp_(0, 1)
        self.save_picture(inp)
    def save_picture(self, tensor):
        unloader = transforms.ToPILImage()
        image = tensor.clone()
        image = image.squeeze(0)
        image = unloader(image)
        image.save("result.jpg")
