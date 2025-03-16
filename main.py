import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torchvision.transforms as transforms

from diffusers import ControlNetModel
from diffusers import DDIMScheduler, AutoencoderKL, ControlNetModel#, StableDiffusionControlNetPipeline, LCMScheduler
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetInpaintPipeline
from controlnet_aux import LineartDetector, CannyDetector, HEDdetector

from ip_adapter.resampler import Resampler
from uniportrait.curricular_face.backbone import get_model
from uniportrait.resampler import UniPortraitFaceIDResampler
from uniportrait.uniportrait_attention_processor import UniPortraitCNAttnProcessor2_0 as UniPortraitCNAttnProcessor
from uniportrait.uniportrait_attention_processor import UniPortraitLoRAAttnProcessor2_0 as UniPortraitLoRAAttnProcessor
from uniportrait.uniportrait_attention_processor import UniPortraitLoRAIPAttnProcessor2_0 as UniPortraitLoRAIPAttnProcessor
from uniportrait.uniportrait_attention_processor import attn_args
from uniportrait.StableDiffusionInpaintPipeline  import StableDiffusionControlNetInpaintPipeline
from uniportrait.utils import prepare_single_faceid_cond_kwargs, process_faceid_image, pad_np_bgr_image, dlib_process

from insightface.app import FaceAnalysis
import PIL
from PIL import Image
import math
import numpy as np
import dlib
import cv2
from tqdm import tqdm
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import LCMScheduler
import cv2
import torch
import numpy as np
from PIL import Image
import pandas as pd


def resize_img(input_image, max_side=1280, min_side=1024, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

def prepare_average_embeding(face_list):
    face_emebdings = []
    for face_path in face_list:
      face_image = load_image(face_path)
      face_image = resize_img(face_image)
      face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
      face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
      face_emb = face_info['embedding']
      face_emebdings.append(face_emb)

    return sum(face_emebdings) / len(face_emebdings)

def prepareMaskAndPoseAndControlImage(pose_image, face_info, padding = 50, mask_grow = 20, resize = True):
    if padding < mask_grow:
        raise ValueError('mask_grow cannot be greater than padding')

    kps = face_info['kps']
    width, height = pose_image.size

    x1, y1, x2, y2 = face_info['bbox']
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # check if image can contain padding
    p_x1 = padding if x1 - padding > 0 else x1
    p_y1 = padding if y1 - padding > 0 else y1
    p_x2 = padding if x2 + padding < width else width - x2
    p_y2 = padding if y2 + padding < height else height - y2

    p_x1, p_y1, p_x2, p_y2 = int(p_x1), int(p_y1), int(p_x2), int(p_y2)

    numpy_array = np.array(pose_image)
    index_y1 = y1 - p_y1; index_y2 = y2 + p_y2
    index_x1 = x1 - p_x1; index_x2 = x2 + p_x2

    # cut the face with paddings
    img = numpy_array[index_y1:index_y2, index_x1:index_x2]

    img = Image.fromarray(img.astype(np.uint8))
    original_width, original_height = img.size

    # mask
    mask = np.array(img)
    mask[:, :] = 0

    m_px1 =  p_x1 - mask_grow if p_x1 - mask_grow > 0 else 0
    m_py1 =  p_y1 - mask_grow if p_y1 - mask_grow > 0 else 0
    m_px2 =  mask_grow if original_width - p_x2 + mask_grow < original_width  else original_width - p_x2
    m_py2 =  mask_grow if original_height - p_y2 + mask_grow < original_height else original_height - p_y2

    mask[
        m_py1:(original_height - p_y2 + m_py2),
        m_px1:(original_width - p_x2 + m_px2)
    ] = 255

    mask = Image.fromarray(mask.astype(np.uint8))

    # resize image and KPS
    kps -= [index_x1, index_y1]
    if resize:
        mask = resize_img(mask)
        img = resize_img(img)
        new_width, new_height = img.size
        kps *= [new_width / original_width, new_height / original_height]
    control_image = draw_kps(img, kps)

    # (mask, pose, control), (original positon of face with padding: x, y, w, h)
    return (mask, img, control_image), (index_x1, index_y1, original_width, original_height)

class ImageProjModel(nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds  # b, c
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens,
                                                              self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class UniPortraitPipeline:

    def __init__(self,
                image_encoder_path = "models/IP-Adapter/models/image_encoder",
                ip_ckpt = "models/IP-Adapter/models/ip-adapter_sd15.bin",
                uniportrait_faceid_ckpt = "models/uniportrait-faceid_sd15.bin",
                uniportrait_router_ckpt = "models/uniportrait-router_sd15.bin",
                face_backbone_ckpt = "models/glint360k_curricular_face_r101_backbone.bin",
    ):

        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.uniportrait_faceid_ckpt = uniportrait_faceid_ckpt
        self.uniportrait_router_ckpt = uniportrait_router_ckpt
        self.face_backbone_ckpt = face_backbone_ckpt
        self.faceapp = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=["detection"])
        self.faceapp.prepare(ctx_id=0, det_size=(448, 448))#640, 640

        self.num_ip_tokens = 4
        self.num_faceid_tokens = 16
        self.lora_rank = 128

        self.device = torch.device("cuda")
        self.torch_dtype = torch.float16

        # load SD pipeline
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=self.torch_dtype)

        controlnet_model_list = []

        unet = UNet2DConditionModel.from_pretrained(
                    'stable-diffusion-v1-5/stable-diffusion-v1-5', 
                    subfolder='unet', 
                    torch_dtype=torch_dtype,)
        controlnet = ControlNetModel.from_unet(unet).to(dtype=torch_dtype)
        del unet
        
        controlnet_ckpt = torch.load('models/instantid-components/controlnet.ckpt', map_location='cpu')
        controlnet.load_state_dict(controlnet_ckpt['state_dict'])
        controlnet = controlnet.to(torch_dtype=torch.float16, device_map="auto").cuda()
        controlnet_model_list.append(controlnet)

        controlnet_model_list.append(ControlNetModel.from_pretrained(f'lllyasviel/control_v11p_sd15_lineart', torch_dtype=torch.float16))

        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            controlnet=controlnet_model_list,#
            torch_dtype=self.torch_dtype,
            scheduler=noise_scheduler,
            feature_extractor=None,
            safety_checker=None,
            vae=vae,
        )
        del controlnet_model_list
        self.pipe.to(self.device)

        self.pipe.unet.requires_grad_(False)
        self.pipe.unet.eval()
        self.pipe.controlnet.requires_grad_(False)
        self.pipe.controlnet.eval()
        self.pipe.text_encoder.requires_grad_(False)    
        self.pipe.vae.requires_grad_(False)


        # load clip image encoder
        self.clip_image_processor = CLIPImageProcessor(size={"shortest_edge": 224}, do_center_crop=False,
                                                       use_square_size=True)
        self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=self.torch_dtype)
        # load face backbone
        self.facerecog_model = get_model("IR_101")([112, 112])
        self.facerecog_model.load_state_dict(torch.load(self.face_backbone_ckpt, map_location="cpu"))
        self.facerecog_model = self.facerecog_model.to(self.device, dtype=self.torch_dtype)
        self.facerecog_model.eval()
        # image proj model
        self.image_proj_model = self.init_image_proj()
        # faceid proj model
        self.faceid_proj_model = self.init_faceid_proj()
        self.faceid_proj_model.eval()
        # set uniportrait and ip adapter
        self.set_uniportrait_and_ip_adapter()
        # load uniportrait and ip adapter
        self.load_uniportrait_and_ip_adapter()

    def init_image_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.clip_image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_ip_tokens,
        ).to(self.device, dtype=self.torch_dtype)
        return image_proj_model

    def init_faceid_proj(self):
        faceid_proj_model = UniPortraitFaceIDResampler(
            intrinsic_id_embedding_dim=512,
            structure_embedding_dim=64 + 128 + 256 + self.clip_image_encoder.config.hidden_size,
            num_tokens=16, depth=6,
            dim=self.pipe.unet.config.cross_attention_dim, dim_head=64,
            heads=12, ff_mult=4,
            output_dim=self.pipe.unet.config.cross_attention_dim
        ).to(self.device, dtype=self.torch_dtype)

        self.image_proj_model = Resampler(
            dim=1280,
            dim_head=64,
            heads=20,
            depth=4,
            num_queries=16,
            embedding_dim=512,
            output_dim=self.net.config.cross_attention_dim,
            ff_mult=4,).to(self.device, dtype=self.torch_dtype).eval()
        self.image_proj_model_in_features = image_emb_dim

        return faceid_proj_model

    def set_uniportrait_and_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = UniPortraitLoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=self.lora_rank,
                ).to(self.device, dtype=self.torch_dtype).eval()
            else:
                attn_procs[name] = UniPortraitLoRAIPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=self.lora_rank,
                    num_ip_tokens=self.num_ip_tokens,
                    num_faceid_tokens=self.num_faceid_tokens,
                ).to(self.device, dtype=self.torch_dtype).eval()
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, ControlNetModel):
                self.pipe.controlnet.set_attn_processor(
                    UniPortraitCNAttnProcessor(
                        num_ip_tokens=self.num_ip_tokens,
                        num_faceid_tokens=self.num_faceid_tokens,
                    )
                )
            elif isinstance(self.pipe.controlnet, MultiControlNetModel):
                for module in self.pipe.controlnet.nets:
                    module.set_attn_processor(
                        UniPortraitCNAttnProcessor(
                            num_ip_tokens=self.num_ip_tokens,
                            num_faceid_tokens=self.num_faceid_tokens,
                        )
                    )
            else:
                raise ValueError

    def load_uniportrait_and_ip_adapter(self):
        
        if self.ip_ckpt:
            ctrl_proj_ckpt = torch.load(self.ctr_proj_path, map_location='cpu')
            self.ctrl_proj_model.load_state_dict(ctrl_proj_ckpt['state_dict'])

        if self.ip_ckpt:
            print(f"loading from {self.ip_ckpt}...")
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
            self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
            ip_layers = nn.ModuleList(self.pipe.unet.attn_processors.values())
            ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

        if self.uniportrait_faceid_ckpt:
            print(f"loading from {self.uniportrait_faceid_ckpt}...")
            state_dict = torch.load(self.uniportrait_faceid_ckpt, map_location="cpu")
            self.faceid_proj_model.load_state_dict(state_dict["faceid_proj"], strict=True)
            ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
            ip_layers.load_state_dict(state_dict["faceid_adapter"], strict=False)

            if self.uniportrait_router_ckpt:
                print(f"loading from {self.uniportrait_router_ckpt}...")
                state_dict = torch.load(self.uniportrait_router_ckpt, map_location="cpu")
                router_state_dict = {}
                for k, v in state_dict["faceid_adapter"].items():
                    if "lora." in k:
                        router_state_dict[k.replace("lora.", "multi_id_lora.")] = v
                    elif "router." in k:
                        router_state_dict[k] = v
                ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
                ip_layers.load_state_dict(router_state_dict, strict=False)

    @torch.inference_mode()
    def get_ip_embeds(self, pil_ip_image):
        ip_image = self.clip_image_processor(images=pil_ip_image, return_tensors="pt", do_rescale=False).pixel_values
        ip_image = ip_image.to(self.device, dtype=self.torch_dtype)  # (b, 3, 224, 224), values being normalized
        ip_embeds = self.clip_image_encoder(ip_image).image_embeds
        ip_prompt_embeds = self.image_proj_model(ip_embeds)
        uncond_ip_prompt_embeds = self.image_proj_model(torch.zeros_like(ip_embeds))
        return ip_prompt_embeds, uncond_ip_prompt_embeds

    @torch.inference_mode()
    def get_single_faceid_embeds(self, pil_face_images, face_structure_scale):
        face_clip_image = self.clip_image_processor(images=pil_face_images, return_tensors="pt", do_rescale=False).pixel_values
        face_clip_image = face_clip_image.to(self.device, dtype=self.torch_dtype)  # (b, 3, 224, 224)
        face_clip_embeds = self.clip_image_encoder(
            face_clip_image, output_hidden_states=True).hidden_states[-2][:, 1:]  # b, 256, 1280

        OPENAI_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device,
                                        dtype=self.torch_dtype).reshape(-1, 1, 1)
        OPENAI_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device,
                                       dtype=self.torch_dtype).reshape(-1, 1, 1)
        facerecog_image = face_clip_image * OPENAI_CLIP_STD + OPENAI_CLIP_MEAN  # [0, 1]
        facerecog_image = torch.clamp((facerecog_image - 0.5) / 0.5, -1, 1)  # [-1, 1]
        facerecog_image = F.interpolate(facerecog_image, size=(112, 112), mode="bilinear", align_corners=False)
        facerecog_embeds = self.facerecog_model(facerecog_image, return_mid_feats=True)[1]

        face_intrinsic_id_embeds = facerecog_embeds[-1]  # (b, 512, 7, 7)
        face_intrinsic_id_embeds = face_intrinsic_id_embeds.flatten(2).permute(0, 2, 1)  # b, 49, 512

        facerecog_structure_embeds = facerecog_embeds[:-1]  # (b, 64, 56, 56), (b, 128, 28, 28), (b, 256, 14, 14)
        facerecog_structure_embeds = torch.cat([
            F.interpolate(feat, size=(16, 16), mode="bilinear", align_corners=False)
            for feat in facerecog_structure_embeds], dim=1)  # b, 448, 16, 16
        facerecog_structure_embeds = facerecog_structure_embeds.flatten(2).permute(0, 2, 1)  # b, 256, 448
        face_structure_embeds = torch.cat([facerecog_structure_embeds, face_clip_embeds], dim=-1)  # b, 256, 1728

        uncond_face_clip_embeds = self.clip_image_encoder(
            torch.zeros_like(face_clip_image[:1]), output_hidden_states=True).hidden_states[-2][:, 1:]  # 1, 256, 1280
        uncond_face_structure_embeds = torch.cat(
            [torch.zeros_like(facerecog_structure_embeds[:1]), uncond_face_clip_embeds], dim=-1)  # 1, 256, 1728

        faceid_prompt_embeds = self.faceid_proj_model(
            face_intrinsic_id_embeds.flatten(0, 1).unsqueeze(0),
            face_structure_embeds.flatten(0, 1).unsqueeze(0),
            structure_scale=face_structure_scale,
        )  # [b, 16, 768]

        uncond_faceid_prompt_embeds = self.faceid_proj_model(
            torch.zeros_like(face_intrinsic_id_embeds[:1]),
            uncond_face_structure_embeds,
            structure_scale=face_structure_scale,
        )  # [1, 16, 768]

        return faceid_prompt_embeds, uncond_faceid_prompt_embeds

    def generate(
            self,
            prompt="A face",
            negative_prompt="nsfw",
            pil_faceid_image=None,
            pil_mask_image = None,
            pil_init_img = None,
            pil_ip_image = None,
            # cond_faceids=None,
            face_structure_scale=1,
            seed=2147483647,
            guidance_scale=7.5,
            num_inference_steps=25,
            pil_mask_image_wo_teeth= None,
            zT=None,
            **kwargs,
    ):
        """
        Args:
            prompt:
            negative_prompt:
            pil_ip_image:
            cond_faceids: [
                {
                    "refs": [PIL.Image] or PIL.Image,
                    (Optional) "mix_refs": [PIL.Image],
                    (Optional) "mix_scales": [float],
                },
                ...
            ]
            face_structure_scale:
            seed:
            guidance_scale:
            num_inference_steps:
            zT:
            **kwargs:
        Returns:
        """

        )

        negative_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry"
        toTensor = transforms.ToTensor()
        process_faceid_image(pil_faceid_image, self.faceapp)
        
        init_emb = toTensor(process_faceid_image(pil_init_img, self.faceapp)).unsqueeze(0)
        mk_emb = toTensor(process_faceid_image(pil_faceid_image, self.faceapp)).unsqueeze(0)
        
        ctrl_prompt_embeds = self.ctrl_proj_model(mk_emb)



        attn_args.reset()
        attn_args.lora_scale = 1.0 # 1.0 if len(cond_faceids) == 1 else 0.0
        attn_args.multi_id_lora_scale = 0 #1.0 if len(cond_faceids) > 1 else 0.0
        attn_args.faceid_scale = 1 #faceid_scale if len(cond_faceids) > 0 else 0.0
        attn_args.num_faceids = 1 #len(cond_faceids)
        h, w = 512, 512#pil_faceid_image.size##256, 256

        prompt = [prompt]  * 1
        negative_prompt = [negative_prompt] * 1

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=True,
                negative_prompt=negative_prompt)
            num_prompts = prompt_embeds.shape[0]

            
            if pil_ip_image is not None:
                ip_prompt_embeds, uncond_ip_prompt_embeds = self.get_ip_embeds(pil_ip_image)
                ip_prompt_embeds = ip_prompt_embeds.repeat(num_prompts, 1, 1)
                uncond_ip_prompt_embeds = uncond_ip_prompt_embeds.repeat(num_prompts, 1, 1)
            else:
                ip_prompt_embeds = uncond_ip_prompt_embeds = \
                    torch.zeros_like(prompt_embeds[:, :1]).repeat(1, self.num_ip_tokens, 1)

            prompt_embeds = torch.cat([prompt_embeds, ip_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_ip_prompt_embeds], dim=1)

            faceid_prompt_embeds, uncond_faceid_prompt_embeds = \
                    self.get_single_faceid_embeds(init_emb, face_structure_scale)
            faceid_mix_prompt_embeds, _ = self.get_single_faceid_embeds(mk_emb, face_structure_scale=0.2)
            faceid_prompt_embeds = faceid_mix_prompt_embeds#*0. + faceid_prompt_embeds*0.1  
                # faceid_prompt_embeds = torch.cat(all_faceid_prompt_embeds, dim=1)
                # uncond_faceid_prompt_embeds = torch.cat(all_uncond_faceid_prompt_embeds, dim=1)

            faceid_prompt_embeds = torch.cat([faceid_prompt_embeds], dim=1)
            uncond_faceid_prompt_embeds = torch.cat([uncond_faceid_prompt_embeds], dim=1)

            faceid_prompt_embeds = faceid_prompt_embeds.repeat(num_prompts, 1, 1)
            uncond_faceid_prompt_embeds = uncond_faceid_prompt_embeds.repeat(num_prompts, 1, 1)

            prompt_embeds = torch.cat([prompt_embeds, faceid_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_faceid_prompt_embeds], dim=1)

            # print(faceid_prompt_embeds.shape)
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        
        mask_strength = 0.9 #1
        pil_init_img = pil_init_img.resize((h,w), Image.BILINEAR)
        init_img = toTensor(pil_init_img).unsqueeze(0).to(self.device, dtype=self.torch_dtype).requires_grad_(False)


        pil_mask_image = pil_mask_image.resize((h,w), Image.BILINEAR)
        pil_mask_image = (np.array(pil_mask_image)>0).astype(np.uint8)*255
        pil_mask_image = Image.fromarray(pil_mask_image)
        mask_image = toTensor(pil_mask_image).unsqueeze(0).to(self.device, dtype=self.torch_dtype).requires_grad_(False)
        
        kps_img = dlib_process(np.array(pil_init_img), dlib_detector, dlib_predictor)
        kps_img = Image.fromarray(kps_img)
        kps_img = kps_img.resize((h,w), Image.BILINEAR)
        kps_img = toTensor(kps_img).unsqueeze(0).to(self.device, dtype=self.torch_dtype).requires_grad_(False)



        lineart_processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
        lineart_img = lineart_processor(input_image = pil_init_img.resize((512,512), Image.BILINEAR))
        lineart_img = lineart_img.resize((h,w), Image.BILINEAR)

        canny = cv2.Canny(np.array(pil_init_img), 50, 200)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        canny = Image.fromarray(canny)
        canny.save('./output/canny.jpg')

        if mask_image.shape[1] > 1:
            mask_image = (mask_image.mean(dim=1, keepdim=True) > 0).to(self.device, dtype=self.torch_dtype)


        images, _, _  = self.pipe( 
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=4.5,
                num_inference_steps=int(math.ceil(20 / 0.9)),
                generator=generator,
                latents=None,
                ctrl_prompt_embeds = ctrl_prompt_embeds

                control_image= [kps_img, lineart_img], #[]
                image= init_img, #, #,
                mask_image =  mask_image,#.to(self.device, dtype=self.torch_dtype),
                strength=0.9,
                controlnet_conditioning_scale=[1,1],
                return_dict = False, 
                )
        return images

if __name__ == "__main__":
    global dlib_detector, dlib_predictor
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
    uniportrait_pipeline = UniPortraitPipeline()
    root = '/data3/rusiru.thushara/makeup/UniPortrait/data/FFHQ/'

    df = pd.read_csv(f'{root}/inference.csv')
    outputs = []
    for source, target in tqdm(zip(df['Source'],df['Target'])):
        id_img = load_image(f'{root}/Source/{source}').resize((512,512), Image.Resampling.BILINEAR)
        mask_S = load_image(f'{root}/mask_S/{source}').resize((512,512), Image.Resampling.BILINEAR)
        id_array = np.array(id_img)
        mask_array = np.array(mask_S).astype(np.uint8)
        mask_array = (mask_array > 0.5).astype(np.uint8)
        result_array = (id_array * mask_array).astype(np.uint8)
        id_img = Image.fromarray(result_array)

        mask = load_image(f'{root}/mask_T/{target}').resize((512,512), Image.Resampling.BILINEAR)
        init_img = load_image(f'{root}/Target/{target}').resize((512,512), Image.Resampling.BILINEAR)
        
        output = uniportrait_pipeline.generate(pil_faceid_image =id_img, pil_mask_image = mask, pil_mask_image_wo_teeth= None, pil_init_img = init_img, pil_ip_image = None, num_inference_steps =25)# pil_faceid_mix_image_1=mk_img, mix_scale_1=0.5,
        output[0].save(f'{root}/{target}')
