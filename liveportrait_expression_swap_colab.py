"""LivePortrait Expression Swap — Free Colab Demo
Based on KwaiVGI LivePortrait | Demo notebook for Face2FaceAI (face2faceai.com)
"""

import torch
import sys
import os

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU runtime not detected.\n"
        "Please enable GPU before running: Runtime → Change runtime type → T4 GPU\n"
        "Then select Runtime → Run all again."
    )
print(f"GPU: {torch.cuda.get_device_name(0)}")

repo_dir = "/content/LivePortrait"
if not os.path.exists(repo_dir):
    print("Cloning KwaiVGI LivePortrait repository...")
    !git clone https://github.com/KwaiVGI/LivePortrait.git {repo_dir}
else:
    print("Repository exists")

os.chdir(repo_dir)
sys.path.insert(0, repo_dir)

# ============================================================================
# PATCH KWAIVGI LIVEPORTRAIT CODE
# ============================================================================
# Patches bypass InsightFace (GPL-licensed) in favor of MediaPipe for face
# detection, and fix model loading to work correctly on Colab GPU.

print("\nPatching cropper.py...")
cropper_path = "src/utils/cropper.py"
with open(cropper_path, 'r') as f:
    code = f.read()

old_import = """from .face_analysis_diy import FaceAnalysisDIY"""
new_import = """# from .face_analysis_diy import FaceAnalysisDIY  # Bypassed"""
code = code.replace(old_import, new_import)

old_import_landmark = """from .human_landmark_runner import LandmarkRunner as HumanLandmark"""
new_import_landmark = """# from .human_landmark_runner import LandmarkRunner as HumanLandmark  # Bypassed"""
code = code.replace(old_import_landmark, new_import_landmark)

old_init = """        self.face_analysis_wrapper = FaceAnalysisDIY(
                    name="buffalo_l",
                    root=self.crop_cfg.insightface_root,
                    providers=face_analysis_wrapper_provider,
                )
        self.face_analysis_wrapper.prepare(ctx_id=device_id, det_size=(512, 512), det_thresh=self.crop_cfg.det_thresh)
        self.face_analysis_wrapper.warmup()"""

new_init = """        # InsightFace bypassed
        self.face_analysis_wrapper = None"""
code = code.replace(old_init, new_init)

old_landmark_init = """        self.human_landmark_runner = HumanLandmark(
            ckpt_path=self.crop_cfg.landmark_ckpt_path,
            onnx_provider=device,
            device_id=device_id,
        )
        self.human_landmark_runner.warmup()"""

new_landmark_init = """        # HumanLandmarkRunner bypassed
        self.human_landmark_runner = None"""
code = code.replace(old_landmark_init, new_landmark_init)

old_crop = """    def crop_source_image(self, img_rgb_: np.ndarray, crop_cfg: CropConfig):
        # crop a source image and get neccessary information
        img_rgb = img_rgb_.copy()  # copy it
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if self.image_type == "human_face":
            src_face = self.face_analysis_wrapper.get(
                img_bgr,
                flag_do_landmark_2d_106=True,
                direction="large-small",
            )
            if len(src_face) == 0:
                log(f"No face detected in the image.")
                return None
            elif len(src_face) > 1:
                log(f"More than one face detected in the image, only pick one face by rule {direction}.")
            src_face = src_face[0]
            lmk = src_face.landmark_2d_106  # this is the 106 landmarks from insightface
            # lmk = landmark_runner.run(img_rgb, lmk)  # NOTE: use a temporal ensemble for landmark fitting
            lmk = self.human_landmark_runner.run(img_rgb, lmk)"""

new_crop = """    def crop_source_image(self, img_rgb_: np.ndarray, crop_cfg: CropConfig, lmk_external=None):
        # crop a source image and get neccessary information
        img_rgb = img_rgb_.copy()  # copy it
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if self.image_type == "human_face":
            if lmk_external is not None:
                lmk = lmk_external
            elif self.face_analysis_wrapper is None:
                log(f"No face detection available")
                return None
            else:
                src_face = self.face_analysis_wrapper.get(
                    img_bgr,
                    flag_do_landmark_2d_106=True,
                    direction="large-small",
                )
                if len(src_face) == 0:
                    log(f"No face detected in the image.")
                    return None
                elif len(src_face) > 1:
                    log(f"More than one face detected in the image, only pick one face by rule {direction}.")
                src_face = src_face[0]
                lmk = src_face.landmark_2d_106
                if self.human_landmark_runner is not None:
                    lmk = self.human_landmark_runner.run(img_rgb, lmk)"""

code = code.replace(old_crop, new_crop)

old_refine = """        # update a 256x256 version for network input
        ret_dct["img_crop_256x256"] = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
        if self.image_type == "human_face":
            lmk = self.human_landmark_runner.run(img_rgb, lmk)
            ret_dct["lmk_crop"] = lmk
            ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / crop_cfg.dsize"""

new_refine = """        # update a 256x256 version for network input
        ret_dct["img_crop_256x256"] = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
        if self.image_type == "human_face":
            if lmk_external is None and self.human_landmark_runner is not None:
                lmk = self.human_landmark_runner.run(img_rgb, lmk)
            ret_dct["lmk_crop"] = lmk
            ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / crop_cfg.dsize"""

code = code.replace(old_refine, new_refine)

with open(cropper_path, 'w') as f:
    f.write(code)
print("✓ cropper.py patched")

print("Patching crop.py...")
crop_path = "src/utils/crop.py"
with open(crop_path, 'r') as f:
    code = f.read()
code = code.replace('CV2_INTERP = cv2.INTER_LINEAR', 'CV2_INTERP = cv2.INTER_LANCZOS4')
with open(crop_path, 'w') as f:
    f.write(code)
print("✓ crop.py patched")

print("Patching helper.py...")
helper_path = "src/utils/helper.py"
with open(helper_path, 'r') as f:
    code = f.read()

old_load = """def load_model(ckpt_path, model_config, device, model_type):
    model_params = model_config['model_params'][f'{model_type}_params']

    if model_type == 'appearance_feature_extractor':
        model = AppearanceFeatureExtractor(**model_params).to(device)
    elif model_type == 'motion_extractor':
        model = MotionExtractor(**model_params).to(device)
    elif model_type == 'warping_module':
        model = WarpingNetwork(**model_params).to(device)
    elif model_type == 'spade_generator':
        model = SPADEDecoder(**model_params).to(device)
    elif model_type == 'stitching_retargeting_module':
        # Special handling for stitching model
        model = StitchingRetargetingNetwork(**model_params)
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(device)
        model.eval()
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    return model"""

new_load = """def load_model(ckpt_path, model_config, device, model_type):
    model_params = model_config['model_params'][f'{model_type}_params']

    if model_type == 'appearance_feature_extractor':
        model = AppearanceFeatureExtractor(**model_params)
    elif model_type == 'motion_extractor':
        model = MotionExtractor(**model_params)
    elif model_type == 'warping_module':
        model = WarpingNetwork(**model_params)
    elif model_type == 'spade_generator':
        model = SPADEDecoder(**model_params)
    elif model_type == 'stitching_retargeting_module':
        model = StitchingRetargetingNetwork(**model_params)
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(device)
        model.eval()
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)

    return model"""

code = code.replace(old_load, new_load)

with open(helper_path, 'w') as f:
    f.write(code)
print("✓ helper.py patched")

from huggingface_hub import hf_hub_download
import os

print("\nDownloading essential LivePortrait models...")
os.makedirs("pretrained_weights", exist_ok=True)

kwaivgi_models = [
    "liveportrait/base_models/appearance_feature_extractor.pth",
    "liveportrait/base_models/motion_extractor.pth",
    "liveportrait/base_models/warping_module.pth",
    "liveportrait/base_models/spade_generator.pth",
    "liveportrait/retargeting_models/stitching_retargeting_module.pth"
]

for model_path in kwaivgi_models:
    local_path = os.path.join("pretrained_weights", model_path)
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {model_path}...")
        hf_hub_download(
            repo_id="KwaiVGI/LivePortrait",
            filename=model_path,
            local_dir="pretrained_weights",
            local_dir_use_symlinks=False
        )
    else:
        print(f"✓ {os.path.basename(model_path)} already exists")

print("✓ All essential models ready")

print("\nInstalling dependencies...")
!pip install -q torch --index-url https://download.pytorch.org/whl/cu121
!pip install -q numpy==1.26.4 opencv-python-headless
!pip install -q gradio mediapipe==0.10.14
!pip install -q tyro pyyaml tqdm
!pip install -q facexlib
print("✓ Dependencies installed")

# Dummy i18n kept for compatibility
os.makedirs("gradio_i18n", exist_ok=True)
with open("gradio_i18n/__init__.py", "w") as f:
    f.write("def gettext(text): return text\n_ = gettext\nclass Translate:\n    def __init__(self, *args, **kwargs): pass\n    def __call__(self, text): return text\n    def __enter__(self): return self\n    def __exit__(self, *args): return False")

print("\nImporting KwaiVGI modules...")
import cv2
import numpy as np
from PIL import Image
import gradio as gr

from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_wrapper import LivePortraitWrapper
from src.utils.cropper import Cropper
from src.utils.camera import get_rotation_matrix
from src.utils.crop import prepare_paste_back, paste_back

print("✓ Modules imported")

print("\nInitializing KwaiVGI models...")
args = ArgumentConfig()
inference_cfg = InferenceConfig()
crop_cfg = CropConfig()

live_portrait_wrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
cropper = Cropper(crop_cfg=crop_cfg)

print("✓ All models initialized")

# MediaPipe detector
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

class FaceAnalysisDIY:
    def __init__(self):
        self.mp_fd = mp_face_detection

    def get(self, img):
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_fd.process(img_rgb)

        faces = []
        if results.detections:
            for detection in results.detections:
                bbox_rel = detection.location_data.relative_bounding_box
                x1 = int(bbox_rel.xmin * w)
                y1 = int(bbox_rel.ymin * h)
                x2 = int((bbox_rel.xmin + bbox_rel.width) * w)
                y2 = int((bbox_rel.ymin + bbox_rel.height) * h)

                faces.append({
                    'bbox': np.array([x1, y1, x2, y2], dtype=np.float32),
                    'det_score': detection.score[0]
                })

        faces.sort(key=lambda f: f['bbox'][0])
        return faces

detector = FaceAnalysisDIY()
print("✓ MediaPipe detector ready")

MAX_INPUT_SIZE = 2000

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def resize_to_limit(img, max_size=MAX_INPUT_SIZE):
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img

    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def expand_bbox(bbox, crop_factor, img_shape):
    """Expand detection bbox by crop_factor, keeping it square based on max(w,h)"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    size = max(w, h) * crop_factor
    new_x1 = int(max(0, center_x - size/2))
    new_y1 = int(max(0, center_y - size/2))
    new_x2 = int(min(img_shape[1], center_x + size/2))
    new_y2 = int(min(img_shape[0], center_y + size/2))
    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)

def draw_face_box(img, face, crop_factor):
    """Draw a single green box around the selected face"""
    result = img.copy()
    bbox = expand_bbox(face["bbox"], crop_factor, img.shape)
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return result

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

state = {
    "source_img": None,
    "source_face": None,
    "source_crop_factor": 2.0,

    "target_img": None,
    "target_face": None,
    "target_crop_factor": 2.0,

    "result": None,
}

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class FeatureContainer:
    def __init__(self, x_s_info, f_s, x_s):
        self.x_s_info = x_s_info
        self.f_s = f_s
        self.x_s = x_s

def extract_features_from_face(face_crop):
    """Extract LivePortrait features from a 256x256 face crop"""
    try:
        if face_crop.shape[:2] != (256, 256):
            face_crop = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        img_tensor = torch.from_numpy(face_crop).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(live_portrait_wrapper.device)

        x_s_info = live_portrait_wrapper.get_kp_info(img_tensor)
        f_s = live_portrait_wrapper.extract_feature_3d(img_tensor)
        x_s = live_portrait_wrapper.transform_keypoint(x_s_info)

        return FeatureContainer(x_s_info, f_s, x_s)

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

# ============================================================================
# SOURCE / TARGET IMAGE HANDLERS
# ============================================================================

def detect_single_face(img):
    """Detect the leftmost face in the image, or None"""
    faces = detector.get(img)
    if not faces:
        return None
    return faces[0]  # leftmost (sorted by x in FaceAnalysisDIY.get)

def get_face_crop(img, face, crop_factor):
    """Extract face region from image using expanded bbox, resized to 256x256"""
    bbox = expand_bbox(face["bbox"], crop_factor, img.shape)
    x1, y1, x2, y2 = bbox.astype(int)
    face_region = img[y1:y2, x1:x2]
    return cv2.resize(face_region, (256, 256), interpolation=cv2.INTER_LANCZOS4)

def process_source_upload(img):
    if img is None:
        return None, "Upload source image", gr.update(value=2.0), None, None

    img = resize_to_limit(img, MAX_INPUT_SIZE)
    face = detect_single_face(img)

    state["source_img"] = img
    state["source_face"] = face
    state["source_crop_factor"] = 2.0
    state["result"] = None

    if face is None:
        return img, "No face detected", gr.update(value=2.0), None, None

    display = draw_face_box(img, face, 2.0)
    preview = get_face_crop(img, face, 2.0)
    return display, "✓ Face detected", gr.update(value=2.0), preview, None

def process_target_upload(img):
    if img is None:
        return None, "Upload target image", gr.update(value=2.0), None, None, None

    img = resize_to_limit(img, MAX_INPUT_SIZE)
    face = detect_single_face(img)

    state["target_img"] = img
    state["target_face"] = face
    state["target_crop_factor"] = 2.0
    state["result"] = None

    if face is None:
        return img, "No face detected", gr.update(value=2.0), None, None, ""

    display = draw_face_box(img, face, 2.0)
    preview = get_face_crop(img, face, 2.0)
    return display, "✓ Face detected", gr.update(value=2.0), preview, preview, "✓ No blending (showing target)"

def update_source_crop(crop_factor):
    state["source_crop_factor"] = crop_factor
    if state["source_img"] is None or state["source_face"] is None:
        return None, "", None
    display = draw_face_box(state["source_img"], state["source_face"], crop_factor)
    preview = get_face_crop(state["source_img"], state["source_face"], crop_factor)
    return display, "", preview

def update_target_crop(crop_factor):
    state["target_crop_factor"] = crop_factor
    if state["target_img"] is None or state["target_face"] is None:
        return None, "", None
    display = draw_face_box(state["target_img"], state["target_face"], crop_factor)
    preview = get_face_crop(state["target_img"], state["target_face"], crop_factor)
    return display, "", preview

# ============================================================================
# EXPRESSION BLENDING
# ============================================================================

def copy_expression(source_preview, target_preview, expr_blend, rot_blend):
    """Copy expression from source to target using LivePortrait"""
    if expr_blend == 0.0 and rot_blend == 0.0:
        if target_preview is not None:
            return target_preview, "✓ No blending (showing target)"
        return None, ""

    if source_preview is None or target_preview is None:
        return None, "Upload and detect faces in both source and target"

    if state["source_face"] is None or state["target_face"] is None:
        return None, "No faces detected in source or target"

    try:
        # Extract face regions
        source_crop = get_face_crop(state["source_img"], state["source_face"], state["source_crop_factor"])
        target_crop = get_face_crop(state["target_img"], state["target_face"], state["target_crop_factor"])

        source_psi = extract_features_from_face(source_crop)
        target_psi = extract_features_from_face(target_crop)

        if source_psi is None or target_psi is None:
            return None, "Feature extraction failed"

        source_info = source_psi.x_s_info
        target_info = target_psi.x_s_info

        # Blend rotation
        pitch_blend = target_info['pitch'] + rot_blend * (source_info['pitch'] - target_info['pitch'])
        yaw_blend = target_info['yaw'] + rot_blend * (source_info['yaw'] - target_info['yaw'])
        roll_blend = target_info['roll'] + rot_blend * (source_info['roll'] - target_info['roll'])

        rotation_matrix_blend = get_rotation_matrix(pitch_blend, yaw_blend, roll_blend)

        # Blend expressions
        target_exp_base = target_info['exp'] + target_info['kp']
        source_exp_base = source_info['exp'] + source_info['kp']
        exp_blend_combined = target_exp_base + expr_blend * (source_exp_base - target_exp_base)

        # Build blended keypoints
        x_d_blend = target_info['scale'] * (exp_blend_combined @ rotation_matrix_blend) + target_info['t']

        # Use ONLY target appearance features (expression transfer, not face swap)
        x_s_target = target_psi.x_s
        f_s_target = target_psi.f_s

        # Apply stitching
        x_d_stitched = live_portrait_wrapper.stitching(x_s_target, x_d_blend)

        # Generate result
        result_dict = live_portrait_wrapper.warp_decode(f_s_target, x_s_target, x_d_stitched)
        result_img = live_portrait_wrapper.parse_output(result_dict['out'])[0]

        if isinstance(result_img, torch.Tensor):
            result_img = (result_img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Insert processed face back into full target image
        result_full = state["target_img"].copy()
        bbox = expand_bbox(state["target_face"]["bbox"], state["target_crop_factor"], state["target_img"].shape)
        x1, y1, x2, y2 = bbox.astype(int)
        face_h, face_w = y2 - y1, x2 - x1
        result_face_resized = cv2.resize(result_img, (face_w, face_h), interpolation=cv2.INTER_LANCZOS4)
        result_full[y1:y2, x1:x2] = result_face_resized

        state["result"] = result_full

        # Return 256x256 preview
        result_preview = cv2.resize(result_face_resized, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        return result_preview, f"✓ Expression copied (expr:{expr_blend:.2f} rot:{rot_blend:.2f})"

    except Exception as e:
        return None, f"Error: {str(e)}"

def auto_update_result(expr_blend, rot_blend):
    """Auto-update result after crop changes if sliders are non-zero"""
    source_preview = get_face_crop(state["source_img"], state["source_face"], state["source_crop_factor"]) if state["source_face"] else None
    target_preview = get_face_crop(state["target_img"], state["target_face"], state["target_crop_factor"]) if state["target_face"] else None

    if expr_blend == 0.0 and rot_blend == 0.0:
        if target_preview is not None:
            return target_preview, "✓ No blending (showing target)"
        return None, ""

    return copy_expression(source_preview, target_preview, expr_blend, rot_blend)

def reset_expression_sliders():
    """Reset expression/rotation sliders to neutral"""
    target_preview = get_face_crop(state["target_img"], state["target_face"], state["target_crop_factor"]) if state["target_face"] else None
    status = "✓ No blending (showing target)" if target_preview is not None else ""
    return target_preview, status, 0.0, 0.0

# ============================================================================
# FINALIZATION
# ============================================================================

def finalize(mode):
    if state["result"] is None or state["target_face"] is None:
        return None, "Adjust sliders first"

    try:
        # 512x512 upsampled face crop
        bbox = expand_bbox(state["target_face"]["bbox"], state["target_crop_factor"], state["target_img"].shape)
        x1, y1, x2, y2 = bbox.astype(int)
        face_crop = state["result"][y1:y2, x1:x2]
        upscaled = cv2.resize(face_crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        return upscaled, "✓ Face upsampled to 512x512"

    except Exception as e:
        return None, f"Error: {str(e)}"

# ============================================================================
# GRADIO UI
# ============================================================================

with gr.Blocks(title="LivePortrait Expression Swap") as demo:
    gr.Markdown("# LivePortrait Expression Swap\nCopy facial expressions from a source face onto a target face.\n\n*Demo notebook for [Face2FaceAI](https://face2faceai.com) — the full app offers advanced blending, a template library, and more.*")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Source Face")
            source_input = gr.Image(label="Upload Source", type="numpy")
            source_display = gr.Image(label="Detection", type="numpy", interactive=False)
            source_status = gr.Textbox(label="Status", value="Upload source image")
            source_crop_factor = gr.Slider(
                1.0, 7.0, value=2.0, step=0.1,
                label="Bbox Size",
                interactive=True
            )

        with gr.Column():
            gr.Markdown("## Target Face")
            target_input = gr.Image(label="Upload Target", type="numpy")
            target_display = gr.Image(label="Detection", type="numpy", interactive=False)
            target_status = gr.Textbox(label="Status", value="Upload target image")
            target_crop_factor = gr.Slider(
                1.0, 7.0, value=2.0, step=0.1,
                label="Bbox Size",
                interactive=True
            )

    with gr.Row():
        with gr.Column():
            source_face_preview = gr.Image(label="Source Face (256x256)", interactive=False)
        with gr.Column():
            target_face_preview = gr.Image(label="Target Face (256x256)", interactive=False)
        with gr.Column():
            result_face_preview = gr.Image(label="Result Face (256x256)", interactive=False)

    result_status = gr.Textbox(label="Result Status", value="")

    with gr.Accordion("Blending Controls", open=True):
        expr_blend = gr.Slider(-1, 2, value=0, step=0.05, label="Expression Blend (0=Target, 1=Source)", interactive=True)
        rot_blend = gr.Slider(-1, 2, value=0, step=0.05, label="Head Rotation Blend (0=Target, 1=Source)", interactive=True)
        reset_blend_btn = gr.Button("Reset Blend Sliders", variant="secondary")

    output_mode = gr.Radio([
        "512x512 Upsampled Face",
    ], value="512x512 Upsampled Face", label="Output Mode")

    done = gr.Button("Done — Finalize Output", variant="primary", size="lg")
    final = gr.Image(label="Final Result")
    final_status = gr.Textbox(label="Status")

    # ========================================================================
    # EVENT HANDLERS
    # ========================================================================

    source_input.change(
        process_source_upload,
        inputs=[source_input],
        outputs=[source_display, source_status, source_crop_factor, source_face_preview, result_face_preview]
    )

    target_input.change(
        process_target_upload,
        inputs=[target_input],
        outputs=[target_display, target_status, target_crop_factor, target_face_preview, result_face_preview, result_status]
    )

    source_crop_factor.change(
        update_source_crop,
        inputs=[source_crop_factor],
        outputs=[source_display, source_status, source_face_preview]
    ).then(
        auto_update_result,
        inputs=[expr_blend, rot_blend],
        outputs=[result_face_preview, result_status]
    )

    target_crop_factor.change(
        update_target_crop,
        inputs=[target_crop_factor],
        outputs=[target_display, target_status, target_face_preview]
    ).then(
        auto_update_result,
        inputs=[expr_blend, rot_blend],
        outputs=[result_face_preview, result_status]
    )

    expr_blend.change(
        copy_expression,
        inputs=[source_face_preview, target_face_preview, expr_blend, rot_blend],
        outputs=[result_face_preview, result_status]
    )

    rot_blend.change(
        copy_expression,
        inputs=[source_face_preview, target_face_preview, expr_blend, rot_blend],
        outputs=[result_face_preview, result_status]
    )

    reset_blend_btn.click(
        reset_expression_sliders,
        inputs=None,
        outputs=[result_face_preview, result_status, expr_blend, rot_blend]
    )

    done.click(
        finalize,
        inputs=[output_mode],
        outputs=[final, final_status]
    )

demo.launch(share=True, inline=False)
