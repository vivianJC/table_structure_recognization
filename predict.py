import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from unet import UNet

from torchvision import transforms

def predict_img(net,
                full_img,
                use_dense_crf=True,
                use_gpu=False):

    net.eval()

    img = full_img
    
    height, width = img.size[:2]
    size = (min(500, int(width/4)), min(500,int(height/4)))
    normMean = [0.4948052, 0.48568845, 0.44682974]
    normStd = [0.24580306, 0.24236229, 0.2603115]
    normTransform = transforms.Normalize(normMean, normStd)
    Transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        normTransform
    ])

    Transform_img = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    img = Transform(img)
    img = img.unsqueeze(0)
    
    full_img = Transform_img(full_img)
    
    with torch.no_grad():
        
        vmasks_pred, hmasks_pred = net(img)
        
        mask_v = vmasks_pred.reshape(vmasks_pred.shape[0]*vmasks_pred.shape[1], vmasks_pred.shape[2]*vmasks_pred.shape[3])
        mask_h = hmasks_pred.reshape(hmasks_pred.shape[0]*hmasks_pred.shape[1], hmasks_pred.shape[2]*hmasks_pred.shape[3])
        
        mask_v = np.argmax(mask_v, 0)
        mask_h = np.argmax(mask_h, 0)
                        
        mask_v = mask_v.reshape(vmasks_pred.shape[2], vmasks_pred.shape[3])
        mask_h = mask_h.reshape(hmasks_pred.shape[2], hmasks_pred.shape[3])
    return mask_v, mask_h



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files_v = []
    out_files_h = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files_v.append("{}_V{}".format(pathsplit[0], pathsplit[1]))
            out_files_h.append("{}_H{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files_v, out_files_h

def mask_to_image(mask):
    return Image.fromarray((mask * 100).astype(np.uint8))

if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files_v, out_files_h = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=3)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        mask_v, mask_h = predict_img(net=net,
                           full_img=img,
                           use_gpu=not args.cpu)

        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)

        if not args.no_save:
            mask_v = mask_v.numpy()
            result_v = mask_to_image(mask_v)
            result_v.save(out_files_v[i])
            
            mask_h = mask_h.numpy()
            result_h = mask_to_image(mask_h)
            result_h.save(out_files_h[i])
            
            print("Mask saved to {}".format(out_files_v[i]))
            print("Mask saved to {}".format(out_files_h[i]))
            
            
            