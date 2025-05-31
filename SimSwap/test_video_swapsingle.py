import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.videoswap import video_swap
import os

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    opt = TestOptions().parse()

    start_epoch, epoch_iter = 1, 0
    crop_size = opt.crop_size

    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'

    # Load model and move it to GPU
    model = create_model(opt)
    model.eval()

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640), mode=mode)
    
    with torch.no_grad():
        pic_a = opt.pic_a_path
        img_a_whole = cv2.imread(pic_a)
        img_a_align_crop, _ = app.get(img_a_whole, crop_size)
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # Move input image to GPU
        img_id = img_id.cuda()

        # Create latent id
        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latent_id = model.netArc(img_id_downsample)
        latent_id = F.normalize(latent_id, p=2, dim=1)

        # Perform video swap (ensure video_swap also works with GPU)
        video_swap(opt.video_path, latent_id, model, app, opt.output_path, temp_results_dir=opt.temp_path, 
                   no_simswaplogo=opt.no_simswaplogo, use_mask=opt.use_mask, crop_size=crop_size)
