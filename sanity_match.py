import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image
from zoedepth.utils.misc import get_image_from_url, colorize
import torch

import argparse
from tqdm import tqdm
from pathlib import Path
import glob

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from pprint import pprint

from matplotlib import pyplot as plt

def sanity(args):
    torch.hub.help("intel-isl/MiDaS", args.model_midas, force_reload=True) 
    # torch.hub.help("intel-isl/MiDaS", "DPT_Large", force_reload=True) 

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = "cpu"
    if DEVICE == "cpu":
        print("WARNING: Running on CPU. This will be slow. Check your CUDA installation.")

    print("*" * 20 + " Testing zoedepth " + "*" * 20)
    conf = get_config("zoedepth", "infer")

    print("Config:")
    pprint(conf)

    model = build_model(conf).to(DEVICE)
    model.eval()
       
    print("-"*20 + " Processing indoor scene " + "-"*20)
    
    # try:
    #     folder_path = args.input_directory.split('/')[-3]
    # except:
    #     folder_path = args.input_directory.split('\\')[-3]
    #     pass
    # output_directory = Path(args.output_directory+"/"+folder_path)
    # output_directory.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        panorama = Image.open(args.input_directory).convert("RGB")
        width, height = panorama.size
        devide = 6
        sift = int(width/devide/4)
        # print(f"Found {len(images_path)} images. Saving files to {output_directory}/")
        depths = []
        for index in tqdm(range(devide)):
            left = int(index*width/devide)+sift
            right = int((index+1)*width/devide)+sift
            if right<=width:
                img = panorama.crop((left,0,right,height))
            else:
                imgr = panorama.crop((left,0,width,height))
                imgl = panorama.crop((0,0,sift,height))
                img = Image.new("RGB",(right-left,height))
                img.paste(imgr,(0,0))
                img.paste(imgl,(width-left,0))
            # img = image.resize((512,384),Image.LANCZOS)
            
            file_stem = "/12052023-1348-depth-"+str(index)            
            X = ToTensor()(img)
            X = X.unsqueeze(0).to(DEVICE)
            depth = model.infer(X).cpu().numpy().squeeze()
            if depths == []:
                depths = depth
            else:
                depths = np.concatenate((depths,depth),axis=1)
            img.save(args.output_directory + "/12052023-1348-"+str(index)+"-"+str(int(sift))+".png")
            np.save(args.output_directory + file_stem +"-"+str(int(sift))+ ".npy", depth)
            plt.imsave(args.output_directory + file_stem +"-"+str(int(sift))+".png", depth, cmap='jet_r')
            # plt.imshow(depth,cmap="jet_r")
        np.save(args.output_directory + "/12052023-1348-depth-"+str(int(sift))+".npy", depths)
        plt.imsave(args.output_directory + "/12052023-1348-depth-"+str(int(sift))+".png", depths, cmap='jet_r')
        # plt.imshow(depths,cmap="jet_r")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_midas', help="midas model", default="DPT_BEiT_L_384")
    # parser.add_argument('-i','--input_directory', help="directory to input images", default="input/12052023-1348/*/l.jpg")
    parser.add_argument('-i','--input_directory', help="directory to input images", default="output/match/12052023-1348.jpg")
    parser.add_argument('-o','--output_directory', help="directory to save output", default="output/match")
    
    args = parser.parse_args()
    sanity(args)       
