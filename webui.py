"""WebUI for ml-sharp 3D Gaussian Splat prediction.

A simple Flask-based web interface for uploading images and generating 3DGS PLY files.
"""

from __future__ import annotations

import logging
import os
import sys
import subprocess
import tempfile
import uuid
import shutil
import threading
import time
import gc
from pathlib import Path
import urllib.parse
import traceback

# --- Environment Auto-Fix Start ---
# Check for Windows + CPU-only PyTorch and attempt to fix
try:
    import torch
    if os.name == 'nt' and not torch.cuda.is_available():
        if '+cpu' in torch.__version__:
            print("!" * 80)
            print(f"WARNING: Detected CPU-only PyTorch ({torch.__version__}) on Windows.")
            print("Attempting to automatically reinstall CUDA-enabled PyTorch...")
            print("!" * 80)
            
            try:
                # Uninstall existing torch packages
                subprocess.check_call([
                    sys.executable, "-m", "pip", "uninstall", "-y", 
                    "torch", "torchvision", "torchaudio"
                ])
                
                # Install GPU versions (using stable CUDA 12.1 index for best gsplat compatibility)
                # Note: Using --no-cache-dir to avoid picking up the cached CPU wheel again
                print("Installing CUDA-enabled PyTorch (cu121)... (This may take a while)")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu121",
                    "--no-cache-dir"
                ])
                
                print("\n" + "=" * 80)
                print("SUCCESS: PyTorch reinstalled with CUDA support.")
                print("Please RESTART this application now to use the GPU.")
                print("=" * 80 + "\n")
                sys.exit(0)
                
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Failed to auto-install PyTorch: {e}")
                print("Please manually run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                # Continue anyway, falling back to CPU logic below
except ImportError:
    pass
# --- Environment Auto-Fix End ---

import numpy as np
import imageio.v2 as iio
import imageio_ffmpeg
import torch.nn.functional as F
from PIL import Image
from flask import Flask, jsonify, render_template, request, send_file

from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians
# Imports for SBS Rendering
from sharp.utils import gsplat, camera

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
LOGGER = logging.getLogger(__name__)

# SILENCE WERKZEUG (HTTP LOGS)
try:
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
except Exception:
    pass

# Flask app - use absolute paths for static and template folders
_base_dir = Path(__file__).parent.absolute()
app = Flask(
    __name__,
    static_folder=str(_base_dir / "webui_static"),
    static_url_path="/static",
    template_folder=str(_base_dir / "webui_templates")
)

# Global model cache
_model_cache = {"predictor": None, "device": None}

# Global job store for async video processing
_active_jobs = {}

# Output directory for generated PLY files
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Model URL
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


def get_device() -> torch.device:
    """Get the best available device."""
    if _model_cache["device"] is not None:
        return _model_cache["device"]

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_predictor() -> tuple[RGBGaussianPredictor, torch.device]:
    """Get or create the Gaussian predictor model."""
    if _model_cache["predictor"] is None:
        target_device = torch.device("cpu")
        
        if torch.cuda.is_available():
            target_device = torch.device("cuda")
            try:
                gpu_name = torch.cuda.get_device_name(0)
                LOGGER.info(f"CUDA GPU detected: {gpu_name}")
            except Exception:
                LOGGER.info("CUDA GPU detected (name unknown)")
        elif torch.mps.is_available():
            target_device = torch.device("mps")
            LOGGER.info("Apple MPS acceleration detected.")
        else:
            LOGGER.info("No active GPU detected. Using CPU.")

        LOGGER.info(f"Targeting device for inference: {target_device}")

        LOGGER.info(f"Downloading model from {DEFAULT_MODEL_URL}")
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                DEFAULT_MODEL_URL, 
                progress=True, 
                map_location="cpu"
            )
        except Exception as e:
            LOGGER.error(f"Failed to download/load model checkpoint: {e}")
            raise

        LOGGER.info("Initializing predictor...")
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval()
        
        final_device = torch.device("cpu")
        if target_device.type != "cpu":
            try:
                LOGGER.info(f"Moving model to {target_device}...")
                predictor.to(target_device)
                dummy = torch.zeros(1).to(target_device)
                del dummy
                final_device = target_device
            except RuntimeError as e:
                LOGGER.warning(f"Failed to initialize on {target_device}: {e}.")
                LOGGER.warning("Falling back to CPU mode.")
                predictor.to("cpu")
                final_device = torch.device("cpu")
        else:
            predictor.to("cpu")

        _model_cache["predictor"] = predictor
        _model_cache["device"] = final_device
        LOGGER.info(f"Model successfully loaded and running on: {final_device}")

    return _model_cache["predictor"], _model_cache["device"]


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    use_fp16: bool = False
) -> Gaussians3D:
    """Predict Gaussians from an image."""
    internal_shape = (1536, 1536)

    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    if use_fp16 and device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.float16):
            gaussians_ndc = predictor(image_resized_pt, disparity_factor)
    else:
        gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/test")
def test_viewer():
    """Serve the test viewer page."""
    return render_template("test-viewer.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Generate a 3DGS PLY file from an uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    quality = request.form.get('quality', 'balanced')
    use_fp16 = (quality == 'fast')

    allowed_extensions = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".tiff", ".tif", ".webp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    try:
        unique_id = str(uuid.uuid4())[:8]
        original_stem = Path(file.filename).stem

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        LOGGER.info(f"Processing uploaded file: {file.filename} | Quality: {quality} | FP16: {use_fp16}")

        image, _, f_px = io.load_rgb(tmp_path)
        height, width = image.shape[:2]

        predictor, device = get_predictor()
        gaussians = predict_image(predictor, image, f_px, device, use_fp16=use_fp16)

        output_filename = f"{original_stem}_{unique_id}.ply"
        output_path = OUTPUT_DIR / output_filename
        save_ply(gaussians, f_px, (height, width), output_path)

        LOGGER.info(f"Saved PLY to: {output_path}")
        tmp_path.unlink()

        return jsonify({
            "success": True,
            "filename": output_filename,
            "download_url": f"/download/{output_filename}",
            "view_url": f"/ply/{output_filename}",
        })

    except Exception as e:
        LOGGER.exception("Error during generation")
        return jsonify({"error": str(e)}), 500


def _process_video_job(job_id, tmp_path, original_stem, unique_id, predictor, device, fps, use_fp16):
    """Background worker to process video frames into PLY sequence."""
    reader = None
    try:
        total_frames = 0
        try:
            tmp_reader = iio.get_reader(tmp_path)
            total_frames = tmp_reader.count_frames()
            tmp_reader.close()
            _active_jobs[job_id]['total_frames'] = total_frames
        except Exception:
            _active_jobs[job_id]['total_frames'] = 0

        reader = iio.get_reader(tmp_path)
        
        for i, frame in enumerate(reader):
            if _active_jobs[job_id]['stop_signal']:
                LOGGER.info(f"Job {job_id} stopped by user.")
                _active_jobs[job_id]['status'] = 'stopped'
                break

            try:
                if frame.shape[2] > 3:
                    frame = frame[:, :, :3]
                
                h, w = frame.shape[:2]
                f_px = io.convert_focallength(w, h, 30.0)

                gaussians = predict_image(predictor, frame, f_px, device, use_fp16=use_fp16)

                frame_filename = f"{original_stem}_{unique_id}_f{i:04d}.ply"
                output_path = OUTPUT_DIR / frame_filename
                save_ply(gaussians, f_px, (h, w), output_path)
                
                del gaussians
                
                if i % 2 == 0:
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    elif device.type == 'mps':
                        try:
                            torch.mps.empty_cache()
                            torch.mps.synchronize() 
                        except Exception:
                            pass
                    gc.collect()
                    
            except Exception as e:
                LOGGER.error(f"Error processing frame {i}: {e}")
                continue 

            _active_jobs[job_id]['files'].append(frame_filename)
            _active_jobs[job_id]['processed_frames'] = i + 1
            
            if i % 10 == 0 or i == total_frames - 1:
                LOGGER.info(f"Job {job_id}: Processed frame {i+1} / {total_frames}")

        if not _active_jobs[job_id]['stop_signal']:
            _active_jobs[job_id]['status'] = 'done'

    except Exception as e:
        LOGGER.exception(f"Job {job_id} failed")
        _active_jobs[job_id]['status'] = 'error'
        _active_jobs[job_id]['error_msg'] = str(e)
    finally:
        if reader:
            try:
                reader.close()
            except:
                pass
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except:
                pass


def _process_sbs_video_job(job_id, tmp_path, original_stem, unique_id, predictor, device, fps, use_fp16, opacity_threshold, stereo_offset, brightness_factor):
    """Background worker for SBS 3D Movie generation (Audio extraction -> Render -> Stitch -> Video)."""
    
    # Verify CUDA for rendering
    if device.type != 'cuda':
        _active_jobs[job_id]['status'] = 'error'
        _active_jobs[job_id]['error_msg'] = "SBS Rendering requires a CUDA GPU. CPU/MPS not supported for server-side rendering."
        return
    
    LOGGER.info(f"Job {job_id} Config | Opacity Threshold: {opacity_threshold} | Stereo Offset: {stereo_offset} | Brightness: {brightness_factor}")

    work_dir = OUTPUT_DIR / f"work_{unique_id}"
    work_dir.mkdir(exist_ok=True)
    audio_path = work_dir / "audio.aac"
    
    reader = None
    
    try:
        # 1. Extract Audio
        LOGGER.info(f"Job {job_id}: Extracting audio...")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        # Check if video has audio track first could be useful, but let's try extraction
        # -vn: no video, -acodec copy: copy audio stream
        audio_cmd = [
            ffmpeg_exe, "-y", "-i", str(tmp_path), 
            "-vn", "-acodec", "copy", str(audio_path)
        ]
        has_audio = False
        try:
            # Capture output to suppress console spam, check return code
            subprocess.run(audio_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if audio_path.exists() and audio_path.stat().st_size > 0:
                has_audio = True
        except subprocess.CalledProcessError:
            LOGGER.warning(f"Job {job_id}: No audio track found or extraction failed. Proceeding without audio.")

        # 2. Setup Processing Loop
        try:
            reader = iio.get_reader(tmp_path)
            total_frames = reader.count_frames()
            _active_jobs[job_id]['total_frames'] = total_frames
        except Exception:
            _active_jobs[job_id]['total_frames'] = 0
            # Re-open if count failed
            reader = iio.get_reader(tmp_path)

        # Initialize Renderer
        # Target resolution: 1920x1080 per eye -> 3840x1080 total
        render_w, render_h = 1920, 1080
        
        # Robust Renderer Initialization for Windows
        # NOTE: Using 'linearRGB' triggers the library to convert Linear->sRGB (Gamma correct), which effectively brightens the image.
        # Using 'sRGB' actually passes the raw Linear data which looks dark.
        try:
            renderer = gsplat.GSplatRenderer(color_space="linearRGB")
        except Exception as e:
            err_str = str(e)
            if "DLL load failed" in err_str or "cl" in err_str or "compiler" in err_str.lower():
                raise RuntimeError(
                    "Missing 'gsplat' binaries. This usually means PyTorch/CUDA version mismatch or missing Visual Studio Build Tools. "
                    "Try reinstalling with PyTorch CUDA 12.1 (cu121) to get compatible wheels."
                )
            raise e
        
        # 3. Frame Loop
        png_files = []
        
        for i, frame in enumerate(reader):
            if _active_jobs[job_id]['stop_signal']:
                LOGGER.info(f"Job {job_id} stopped by user.")
                _active_jobs[job_id]['status'] = 'stopped'
                break

            try:
                if frame.shape[2] > 3: frame = frame[:, :, :3]
                
                # Predict
                h, w = frame.shape[:2]
                f_px = io.convert_focallength(w, h, 30.0)
                gaussians = predict_image(predictor, frame, f_px, device, use_fp16=use_fp16)

                # --- HALO REMOVAL LOGIC ---
                if opacity_threshold > 0.0:
                    # Filter out weak, semi-transparent splats that cause "ghosting" around hair/edges
                    mask = gaussians.opacities[0] > opacity_threshold
                    
                    gaussians = Gaussians3D(
                        mean_vectors=gaussians.mean_vectors[:, mask],
                        singular_values=gaussians.singular_values[:, mask],
                        quaternions=gaussians.quaternions[:, mask],
                        colors=gaussians.colors[:, mask],
                        opacities=gaussians.opacities[:, mask]
                    )
                # -------------------------

                # -- Rendering Logic --
                
                # Prepare Intrinsics for 1920x1080 render
                # We scale f_px to match the target render resolution relative to input
                scale_x = render_w / w
                scale_y = render_h / h
                # Use the larger scale to cover FoV or average? Let's assume input fits 
                # roughly into standard view.
                # Actually, standard sharp camera logic:
                f_px_render = f_px * (render_w / w) 

                intrinsics = torch.tensor([
                    [f_px_render, 0, render_w / 2, 0],
                    [0, f_px_render, render_h / 2, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ], device=device, dtype=torch.float32)

                # Prepare Extrinsics (Stereo)
                # Left Eye
                ext_left = torch.eye(4, device=device)
                ext_left[0, 3] = stereo_offset # Translate camera right = scene moves left
                
                # Right Eye
                ext_right = torch.eye(4, device=device)
                ext_right[0, 3] = -stereo_offset 
                
                # Render Left
                out_left = renderer(
                    gaussians, 
                    extrinsics=ext_left.unsqueeze(0), 
                    intrinsics=intrinsics.unsqueeze(0),
                    image_width=render_w, image_height=render_h
                )
                # Apply brightness factor + clamp
                img_left_tensor = (out_left.color[0] * brightness_factor).clamp(0, 1)
                img_left = (img_left_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()

                # Render Right
                out_right = renderer(
                    gaussians, 
                    extrinsics=ext_right.unsqueeze(0), 
                    intrinsics=intrinsics.unsqueeze(0),
                    image_width=render_w, image_height=render_h
                )
                # Apply brightness factor + clamp
                img_right_tensor = (out_right.color[0] * brightness_factor).clamp(0, 1)
                img_right = (img_right_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()

                # Stitch (Side-by-Side)
                # Shape is (H, W, C). Concatenate along Width (axis 1)
                img_sbs = np.concatenate((img_left, img_right), axis=1) # Result: 3840x1080
                
                # Save PNG
                frame_name = f"frame_{i:05d}.png"
                frame_path = work_dir / frame_name
                Image.fromarray(img_sbs).save(frame_path)
                png_files.append(frame_path)

                # Cleanup VRAM immediately
                del gaussians, out_left, out_right, img_left_tensor, img_right_tensor, img_left, img_right, img_sbs, intrinsics, ext_left, ext_right
                
                # Aggressive memory management for long renders
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                LOGGER.error(f"Error processing SBS frame {i}: {e}")
                # If we catch the compile error inside the loop, re-raise to stop job
                if "cl" in str(e) or "DLL" in str(e) or "compiler" in str(e):
                     raise RuntimeError("gsplat compilation failed. Install VS Build Tools or compatible wheels.")
                continue

            _active_jobs[job_id]['processed_frames'] = i + 1
            if i % 10 == 0:
                 LOGGER.info(f"Job {job_id}: SBS Rendered frame {i+1}")

        # 4. Final Assembly (FFmpeg)
        if len(png_files) > 0:
            output_filename = f"{original_stem}_{unique_id}_SBS.mp4"
            output_path = OUTPUT_DIR / output_filename
            
            LOGGER.info(f"Job {job_id}: Encoding SBS video...")
            
            # Input pattern for ffmpeg sequence
            input_pattern = str(work_dir / "frame_%05d.png")
            
            # Construct FFmpeg command
            # -framerate: set input fps
            # -i pattern: input images
            # -i audio: input audio (if exists)
            # -c:v libx264: encoding
            # -pix_fmt yuv420p: compatibility
            # -shortest: finish when shortest stream ends (images usually)
            
            cmd = [
                ffmpeg_exe, "-y",
                "-framerate", str(fps),
                "-i", input_pattern
            ]
            
            if has_audio:
                cmd.extend(["-i", str(audio_path)])
            
            cmd.extend([
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18", # High quality
                "-preset", "fast"
            ])
            
            if has_audio:
                # If we stopped early, -shortest handles trimming audio length
                cmd.append("-shortest")
            
            cmd.append(str(output_path))
            
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                
                # Register the final file in the job so frontend sees it
                _active_jobs[job_id]['files'].append(output_filename)
                
                if not _active_jobs[job_id]['stop_signal']:
                    _active_jobs[job_id]['status'] = 'done'
                
            except subprocess.CalledProcessError as e:
                LOGGER.error(f"FFmpeg encoding failed: {e.stderr.decode()}")
                _active_jobs[job_id]['status'] = 'error'
                _active_jobs[job_id]['error_msg'] = "Video encoding failed."
        else:
             _active_jobs[job_id]['status'] = 'error'
             _active_jobs[job_id]['error_msg'] = "No frames were processed successfully."

    except Exception as e:
        LOGGER.exception(f"Job {job_id} failed")
        _active_jobs[job_id]['status'] = 'error'
        _active_jobs[job_id]['error_msg'] = str(e)
    finally:
        if reader:
            try: reader.close()
            except: pass
        if tmp_path.exists():
            try: tmp_path.unlink()
            except: pass
        # Clean up work directory (PNGs and Audio)
        if work_dir.exists():
            try:
                shutil.rmtree(work_dir)
            except Exception as e:
                LOGGER.warning(f"Failed to cleanup temp dir {work_dir}: {e}")


@app.route("/generate_video", methods=["POST"])
def generate_video():
    """Start async video generation."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Get Quality/FP16 Setting for video
    quality = request.form.get('quality', 'balanced')
    use_fp16 = (quality == 'fast')
    
    # Get Mode: 'ply_seq' (default) or 'sbs_movie'
    output_mode = request.form.get('output_mode', 'ply_seq')
    
    # Get New Settings (Defaults applied if missing)
    opacity_threshold = float(request.form.get('opacity_threshold', 0.0))
    stereo_offset = float(request.form.get('stereo_offset', 0.03))
    brightness_factor = float(request.form.get('brightness_factor', 1.0))

    LOGGER.info(f"Starting video generation | Quality: {quality} | Mode: {output_mode}")

    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    try:
        unique_id = str(uuid.uuid4())[:8]
        original_stem = Path(file.filename).stem
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        # Get metadata
        try:
            reader = iio.get_reader(tmp_path)
            meta = reader.get_meta_data()
            fps = meta.get('fps', 30.0)
            reader.close() 
        except Exception:
            fps = 30.0
        
        # Prepare job
        job_id = str(uuid.uuid4())
        _active_jobs[job_id] = {
            "status": "running",
            "files": [],
            "total_frames": 0,
            "processed_frames": 0,
            "stop_signal": False,
            "error_msg": "",
            "fps": fps,
            "mode": output_mode
        }

        # Get model (ensure loaded)
        predictor, device = get_predictor()
        
        # Select worker based on mode
        if output_mode == 'sbs_movie':
            thread = threading.Thread(
                target=_process_sbs_video_job,
                args=(job_id, tmp_path, original_stem, unique_id, predictor, device, fps, use_fp16, opacity_threshold, stereo_offset, brightness_factor)
            )
        else:
            thread = threading.Thread(
                target=_process_video_job,
                args=(job_id, tmp_path, original_stem, unique_id, predictor, device, fps, use_fp16)
            )

        thread.start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "running"
        })

    except Exception as e:
        LOGGER.exception("Error starting video generation")
        return jsonify({"error": str(e)}), 500


@app.route("/job_status/<job_id>")
def job_status(job_id):
    """Check status of a background job."""
    job = _active_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify({
        "status": job["status"],
        "processed": job["processed_frames"],
        "total": job["total_frames"],
        "files": job["files"], # Returns full list so client can see what's new
        "fps": job["fps"],
        "error": job["error_msg"],
        "mode": job.get("mode", "ply_seq"),
        "base_url": "/download/" if job.get("mode") == "sbs_movie" else "/ply/"
    })


@app.route("/stop_job/<job_id>", methods=["POST"])
def stop_job(job_id):
    """Signal a job to stop."""
    job = _active_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    job["stop_signal"] = True
    return jsonify({"success": True, "status": "stopping"})


@app.route("/scan_local", methods=["POST"])
def scan_local():
    """Scan a local path for PLY files."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400

        path_str = data.get("path")
        mode = data.get("mode") # 'single' or 'video'
        
        if not path_str:
            return jsonify({"error": "No path provided"}), 400

        # Clean path string (remove quotes users might copy-paste)
        path_str = path_str.strip().strip('"').strip("'")
        path = Path(path_str)
        
        if not path.exists():
            return jsonify({"error": f"Path does not exist: {path_str}"}), 404
            
        if mode == "single":
            if not path.is_file() or path.suffix.lower() != ".ply":
                 return jsonify({"error": "Path is not a valid PLY file"}), 400
            
            # Quote path to safely handle spaces in URL
            encoded_path = urllib.parse.quote(str(path))
            return jsonify({
                "success": True,
                "type": "single",
                "view_url": f"/view_local?path={encoded_path}",
                "filename": path.name
            })
            
        elif mode == "video":
            if not path.is_dir():
                 return jsonify({"error": "Path is not a directory"}), 400
            
            ply_files = sorted([f for f in path.glob("*.ply")])
            if not ply_files:
                 return jsonify({"error": "No PLY files found in directory"}), 400
                 
            # Create full URLs
            file_urls = []
            for p in ply_files:
                 encoded_path = urllib.parse.quote(str(p))
                 file_urls.append(f"/view_local?path={encoded_path}")
            
            return jsonify({
                "success": True,
                "type": "video",
                "fps": 30, # Defaulting to 30 as we can't infer from just files easily
                "frame_count": len(file_urls),
                "ply_files": file_urls,
                "base_url": "" # Empty because urls are fully formed
            })
        
        return jsonify({"error": "Invalid mode"}), 400
        
    except Exception as e:
        LOGGER.exception("Error scanning local path")
        # Return JSON error instead of 500 HTML
        return jsonify({"error": f"Server Error: {str(e)}"}), 500


@app.route("/view_local")
def view_local():
    """Serve a file from a local absolute path."""
    try:
        path_str = request.args.get("path")
        if not path_str:
            return "No path provided", 400
        
        # Check if path exists before sending
        path = Path(path_str)
        if not path.exists() or not path.is_file():
            return "File not found", 404

        return send_file(path, mimetype="application/octet-stream")
    except Exception as e:
        LOGGER.exception("Error serving local file")
        return str(e), 500


@app.route("/download/<filename>")
def download(filename: str):
    """Download a generated file."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    # Determine mimetype based on extension
    mime = "application/octet-stream"
    if filename.endswith(".mp4"):
        mime = "video/mp4"

    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename,
        mimetype=mime,
    )


@app.route("/ply/<filename>")
def serve_ply(filename: str):
    """Serve a PLY file for the viewer."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    return send_file(
        file_path,
        mimetype="application/octet-stream",
    )


@app.route("/status")
def status():
    """Get server status."""
    device = get_device()
    model_loaded = _model_cache["predictor"] is not None
    return jsonify({
        "status": "ok",
        "device": str(device),
        "model_loaded": model_loaded,
        "cuda_available": torch.cuda.is_available(),
    })

@app.route("/open_output_folder", methods=["POST"])
def open_output_folder():
    """Opens the 'output' directory in the system file explorer."""
    try:
        # Get the absolute path to the 'output' folder in the app root
        path = str(OUTPUT_DIR.absolute())
        
        if os.name == 'nt':  # Windows
            os.startfile(path)
        elif sys.platform == 'darwin':  # macOS
            subprocess.Popen(['open', path])
        else:  # Linux
            subprocess.Popen(['xdg-open', path])
            
        return jsonify({"success": True})
    except Exception as e:
        LOGGER.error(f"Failed to open folder: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ml-sharp WebUI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--preload", action="store_true", help="Preload model on startup")

    args = parser.parse_args()

    if args.preload:
        LOGGER.info("Preloading model...")
        get_predictor()

    LOGGER.info(f"Starting WebUI at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)