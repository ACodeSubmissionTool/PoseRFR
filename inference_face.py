import os
import math
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace

from ldm.xformers_state import auto_xformers_status
from model.cldm import TwoStreamControlLDM
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts
from utils.image import auto_resize, pad
from utils.file import load_file_from_url
from utils.face_restoration_helper import FaceRestoreHelper
from model.spaced_sampler import SpacedSampler
from dataset.codeformer import CodeformerDataset
import einops
from torch.utils.data import DataLoader
import pickle
import time
from typing import List, Tuple, Optional
from model.cond_fn import MSEGuidance


@torch.no_grad()
def process(
    model: TwoStreamControlLDM,
    control_imgs: List[np.ndarray],
    steps: int,
    strength: float,
    color_fix_type: str,
    disable_preprocess_model: bool,
    cond_fn: Optional[MSEGuidance],
    return_latent = False,
    AFR=False
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply model on a list of low-quality images.
    
    Args:
        model (ControlLDM): Model.
        control_imgs (List[np.ndarray]): A list of low-quality images (HWC, RGB, range in [0, 255]).
        steps (int): Sampling steps.
        strength (float): Control strength. Set to 1.0 during training.
        color_fix_type (str): Type of color correction for samples.
        disable_preprocess_model (bool): If specified, preprocess model (SwinIR) will not be used.
        cond_fn (Guidance | None): Guidance function that returns gradient to guide the predicted x_0.
        AFR (bool): Specified to adopt Adaptive feature refinement. 
    
    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
        stage1_preds (List[np.ndarray]): Outputs of preprocess model (HWC, RGB, range in [0, 255]). 
            If `disable_preprocess_model` is specified, then preprocess model's outputs is the same 
            as low-quality inputs.
    """
    n_samples = len(control_imgs)
    sampler = SpacedSampler(model, var_type="fixed_small")
    control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    time_s = time.time() 
    if not disable_preprocess_model:
        control = model.preprocess_model(control)
    model.control_scales = [strength] * 13
    
    if cond_fn is not None:
        cond_fn.load_target(2 * control - 1)
    
    height, width = control.size(-2), control.size(-1)
    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)

    if AFR:
        samples = sampler.sample_afr(
            steps=steps, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=cond_fn,
            color_fix_type=color_fix_type
        )
    elif return_latent:
        samples,latents = sampler.sample(
                steps=steps, shape=shape, cond_img=control,
                positive_prompt="", negative_prompt="", x_T=x_T,
                cfg_scale=1.0, cond_fn=cond_fn,
                color_fix_type=color_fix_type,
                return_latent=return_latent
            )
        latents = [latents[i] for i in range(n_samples)]
    else:
        samples = sampler.sample(
            steps=steps, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=cond_fn,
            color_fix_type=color_fix_type
        )

    x_samples = samples.clamp(0, 1)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    #control = (einops.rearrange(control, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    preds = [x_samples[i] for i in range(n_samples)]
    sr = [control[i] for i in range(n_samples)]
    if return_latent:
        return preds, sr, latents
    return preds

def check_device(device):
    if device == "cuda":
        # check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled.")
            device = "cpu"
    else:
        # xformers only support CUDA. Disable xformers when using cpu or mps.
        disable_xformers()
        if device == "mps":
            # check if MPS is available
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not "
                        "built with MPS enabled.")
                    device = "cpu"
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine.")
                    device = "cpu"
    print(f'using device {device}')
    return device


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # model
    # Specify the model ckpt path, and the official model can be downloaded direclty.
    parser.add_argument("--ckpt", type=str, help='Model checkpoint.', default='None')
    parser.add_argument("--config", type=str, default='configs/model/cldm_twoS.yaml', help='Model config file.')

    # input and preprocessing
    parser.add_argument("--input", type=str, default='')
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=512, help='Image size as the model input.')
    parser.add_argument("--disable_preprocess_model", action="store_true")
    parser.add_argument("--use_afr", action="store_true")
    parser.add_argument("--generate_latent", action="store_true")
    
    # postprocessing and saving
    parser.add_argument("--color_fix_type", type=str, default="wavelet", choices=["wavelet", "adain", "none"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--show_lq", action="store_true")
    parser.add_argument("--skip_if_exist", action="store_true")
    
    # change seed to finte-tune your restored images! just specify another random number.
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    
    return parser.parse_args()

def build_model(model_config, ckpt):
    ''''
        model_config: model architecture config file.
        ckpt: checkpoint file path of the main model.
    '''
    model = instantiate_from_config(OmegaConf.load(model_config))
    load_state_dict(model, torch.load(ckpt), strict=False)
    model.freeze()
    return model


def main() -> None:
    args = parse_args()
    img_save_ext = 'png'
    pl.seed_everything(args.seed)    
    #assert os.path.isdir(args.input)

    args.device = check_device(args.device)
    model = build_model(args.config, args.ckpt).to(args.device)

    # ------------------ set up FaceRestoreHelper -------------------
    face_helper = FaceRestoreHelper(
        device=args.device, 
        upscale_factor=1, 
        face_size=args.image_size
        )
    
    if args.generate_latent:
        dataset_config = OmegaConf.load("./configs/dataset/face_train_extrem.yaml")
        dataset = instantiate_from_config(dataset_config['dataset'])
        test_dataloader = DataLoader(dataset,batch_size = 4, shuffle = False, num_workers=16, drop_last = False)

        os.makedirs(args.output, exist_ok=True)
        for epoch in range(2):
            for batch_idx, batch in enumerate(test_dataloader):
                # read image
                lq = batch['hint']
                hq = batch['jpg'].permute(0,3,1,2)
                loc_left_eye = batch['loc_left_eye']
                loc_right_eye = batch['loc_right_eye']
                loc_mouth = batch['loc_mouth']
                
                preds, sr,latents = process(
                    model, lq, steps=args.steps,
                    strength=1,
                    color_fix_type=args.color_fix_type,
                    disable_preprocess_model=args.disable_preprocess_model,
                    cond_fn=None, return_latent=True
                )

                for id in range(len(preds)):
                    tmp = {'sr':sr[id].cpu().detach(),'latent':latents[id].cpu().detach(),'hq':hq[id],'pred':preds[id],'loc_left_eye':loc_left_eye[id],'loc_right_eye':loc_right_eye[id],'loc_mouth':loc_mouth[id]}#,'pred':preds[id]}
                    name = '%d_%d_%d.pkl'%(epoch,batch_idx,id)
                    output = os.path.join(args.output, name)
                    with open(output,'wb') as f:
                        pickle.dump(tmp,f)
                return
    

    for file_path in list_image_files(args.input, follow_links=True):
        # read image
        lq = Image.open(file_path).convert("RGB")
        lq_resized = auto_resize(lq, args.image_size)
        x = pad(np.array(lq_resized), scale=64)

        face_helper.clean_all()
        # the input faces are already cropped and aligned
        face_helper.cropped_faces = [x]

        parent_dir, img_basename, _ = get_file_name_parts(file_path)
        rel_parent_dir = os.path.relpath(parent_dir, args.input)
        output_parent_dir = os.path.join(args.output, rel_parent_dir)
        restored_face_dir = os.path.join(output_parent_dir, 'restored_faces')
        os.makedirs(restored_face_dir, exist_ok=True)

        basename =  img_basename
        if os.path.exists(os.path.join(restored_face_dir, f'{basename}.{img_save_ext}')):
            if args.skip_if_exist:
                print(f"Exists, skip face image {basename}...")
                continue
            else:
                raise RuntimeError(f"Image {basename} already exist")
        try:
     
            preds = process(
                model, face_helper.cropped_faces, steps=args.steps,
                strength=1,
                color_fix_type=args.color_fix_type,
                disable_preprocess_model=args.disable_preprocess_model,
                cond_fn=None, AFR=args.use_afr
            )

                
        except RuntimeError as e:
            # Avoid cuda_out_of_memory error.
            print(f"{file_path}, error: {e}")
            continue
        
        for restored_face in preds:
            face_helper.add_restored_face(np.array(restored_face))

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
            # save restored face
            save_face_name = f'{basename}.{img_save_ext}'
            # remove padding
            restored_face = restored_face[:lq_resized.height, :lq_resized.width, :]
            save_restore_path = os.path.join(restored_face_dir, save_face_name)
            Image.fromarray(restored_face).save(save_restore_path)

        print(f"Face image {basename} saved to {output_parent_dir}")


if __name__ == "__main__":
    main()
