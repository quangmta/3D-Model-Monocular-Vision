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
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    torch.hub.help("intel-isl/MiDaS", args.model_midas, force_reload=True) 
    # torch.hub.help("intel-isl/MiDaS", "DPT_Large", force_reload=True) 

    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
    
    try:
        folder_name = args.input_directory.split('/')[-1]
    except:
        folder_name = args.input_directory.split('\\')[-1]
        pass
    folder_name = folder_name[:-4]
    # print(folder_name)
    
    with torch.no_grad():
        panorama = Image.open(args.input_directory).convert("RGB")
        width, height = panorama.size
        # print(panorama.size)
        devide = 6
        shift = 250
        
        new_output_img = args.output_directory+"/"+folder_name+'/'+str(shift)+'/'+'img'
        new_output_depth = args.output_directory+"/"+folder_name+'/'+str(shift)+'/'+'first_depth'
        Path(new_output_img).mkdir(parents=True,exist_ok=True)    
        Path(new_output_depth).mkdir(parents=True,exist_ok=True)
        
        # print(f"Found {len(images_path)} images. Saving files to {output_directory}/")
        depths = []
        for index in tqdm(range(devide)):
            left = int(index*width/devide)+shift
            right = int((index+1)*width/devide)+shift
            if right<=width:
                img = panorama.crop((left,0,right,height))
            else:
                imgr = panorama.crop((left,0,width,height))
                imgl = panorama.crop((0,0,shift,height))
                img = Image.new("RGB",(right-left,height))
                img.paste(imgr,(0,0))
                img.paste(imgl,(width-left,0))
            # print(img.size)
            # img = image.resize((512,384),Image.LANCZOS)
            # print("???")
            file_stem = "/"+folder_name+"-depth-"+str(index)   
            # print(file_stem)         
            X = ToTensor()(img)
            X = X.unsqueeze(0).to(DEVICE)
            # print("start")
            depth = model.infer(X).cpu().numpy().squeeze()
            # print("done")
            if depths == []:
                depths = depth
            else:
                depths = np.concatenate((depths,depth),axis=1)
            img.save(new_output_img +"/"+ folder_name+"-"+str(index)+"-"+str(int(shift))+".png")
            np.save(new_output_depth + file_stem +"-"+str(int(shift))+ ".npy", depth)
            plt.imsave(new_output_img + file_stem +"-"+str(int(shift))+".png", depth, cmap='jet_r')
            # plt.imshow(depth,cmap="jet_r")
        np.save(new_output_depth + "/"+folder_name+"-depth-"+str(int(shift))+".npy", depths)
        plt.imsave(new_output_img + "/"+folder_name+"-depth-"+str(int(shift))+".png", depths, cmap='jet_r')
        # plt.imshow(depths,cmap="jet_r")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_midas', help="midas model", default="DPT_BEiT_L_384")
    # parser.add_argument('-i','--input_directory', help="directory to input images", default="input/12052023-1348/*/l.jpg")
    parser.add_argument('-i','--input_directory', help="directory to input images", default="output/panorama/27072023-1628.png")
    parser.add_argument('-o','--output_directory', help="directory to save output", default="output")
    
    args = parser.parse_args()
    sanity(args)       
