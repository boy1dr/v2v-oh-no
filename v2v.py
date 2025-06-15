import argparse
import os
import shutil
import subprocess
import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path

# --- SCRIPT CONFIGURATION ---
# Use a GPU if available, otherwise fallback to CPU. This is crucial for performance.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Define temporary directories for the intermediate files
FRAME_DIR = "temp_frames"
OUTPUT_DIR = "temp_output_frames"
TEMP_AUDIO_FILE = "temp_audio.wav"
# Fixed seed for reproducible image generation
GENERATION_SEED = 42

def extract_frames(video_path, fps):
    """
    Converts a video into a sequence of images.
    
    Args:
        video_path (str): Path to the input video file.
        fps (int): The number of frames to extract per second.
    """
    print(f"[*] Step 1: Extracting frames from '{video_path}' at {fps} FPS...")
    
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR)

    video = cv2.VideoCapture(video_path)
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    saved_frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        
        if frame_count % frame_interval == 0:
            # Save frame as a PNG file. Using a consistent naming scheme is important.
            frame_filename = os.path.join(FRAME_DIR, f"frame_{saved_frame_count:05d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
            
        frame_count += 1
        
    video.release()
    print(f"[+] Successfully extracted {saved_frame_count} frames to '{FRAME_DIR}'.")
    return saved_frame_count

def extract_audio(video_path):
    """
    Extracts audio from the input video file.
    
    Args:
        video_path (str): Path to the input video file.
        
    Returns:
        bool: True if audio was successfully extracted, False otherwise.
    """
    print(f"[*] Extracting audio from '{video_path}'...")
    
    try:
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # Use PCM codec for compatibility
            '-ar', '44100',  # Sample rate
            '-ac', '2',  # Stereo
            '-y',  # Overwrite output file
            TEMP_AUDIO_FILE
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[+] Audio extracted successfully to '{TEMP_AUDIO_FILE}'")
            return True
        else:
            print(f"[!] No audio track found or failed to extract audio: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("[!] ffmpeg not found. Please install ffmpeg to enable audio support.")
        print("    Ubuntu: sudo apt install ffmpeg")
        return False
    except Exception as e:
        print(f"[!] Error extracting audio: {e}")
        return False

def caption_frames(num_frames):
    """
    Generates a detailed text description for each extracted frame.
    
    Args:
        num_frames (int): The total number of frames to process.
        
    Returns:
        list: A list of text captions, one for each frame.
    """
    print(f"[*] Step 2: Generating captions for {num_frames} frames...")
    
    # Load the image captioning model (Salesforce BLIP)
    # This model provides a good balance of speed and descriptive quality.
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE)
    
    captions = []
    for i in range(num_frames):
        frame_path = os.path.join(FRAME_DIR, f"frame_{i:05d}.png")
        raw_image = Image.open(frame_path).convert("RGB")
        
        # Prepare the image for the model
        inputs = caption_processor(raw_image, return_tensors="pt").to(DEVICE)
        
        # Generate the caption
        out = caption_model.generate(**inputs, max_new_tokens=50) # Increase max_new_tokens for more detail
        caption = caption_processor.decode(out[0], skip_special_tokens=True)
        
        # We can add more context to the prompt for better results
        # This is a good place for "prompt engineering"
        enhanced_prompt = f"A cinematic, high-detail photograph of {caption}, intricate details, hyperrealistic."
        captions.append(enhanced_prompt)
        
        print(f"  - Frame {i+1}/{num_frames}: {enhanced_prompt}")
        
    print(f"[+] Successfully generated {len(captions)} captions.")
    return captions

def generate_images_from_captions(captions):
    """
    Generates new images based on the text captions using a text-to-image model.
    Optimized for RTX 3080 10GB VRAM with fixed seed for reproducible results.
    
    Args:
        captions (list): A list of text prompts.
    """
    print(f"[*] Step 3: Generating new images from {len(captions)} captions...")
    print(f"[*] Using fixed seed: {GENERATION_SEED}")

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # Import the specific Stable Diffusion pipeline
    from diffusers import StableDiffusionPipeline
    
    # Load the Stable Diffusion pipeline with VRAM optimizations
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,  # Always use float16 for VRAM savings
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True
    )
    
    # Apply VRAM optimizations for RTX 3080
    if DEVICE == "cuda":
        # Enable attention slicing to reduce VRAM usage
        pipe.enable_attention_slicing(1)
        
        # Enable CPU offloading to use system RAM when needed
        pipe.enable_model_cpu_offload()
        
        # Enable sequential CPU offload for even lower VRAM usage
        # pipe.enable_sequential_cpu_offload()  # Uncomment if still having VRAM issues
    else:
        pipe = pipe.to(DEVICE)

    for i, prompt in enumerate(captions):
        print(f"  - Generating image {i+1}/{len(captions)}...")
        
        # Clear GPU cache before each generation
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        try:
            # Generate image with VRAM-friendly parameters and fixed seed
            generator = torch.Generator(device=DEVICE).manual_seed(GENERATION_SEED + i)
            image = pipe(
                prompt, 
                num_inference_steps=20,  # Reduced for faster generation
                guidance_scale=7.5,
                height=512,
                width=512,
                generator=generator  # Fixed seed for reproducible results
            ).images[0]
            
            output_filename = os.path.join(OUTPUT_DIR, f"output_frame_{i:05d}.png")
            image.save(output_filename)
            
        except torch.cuda.OutOfMemoryError:
            print(f"  [!] VRAM error on frame {i+1}, trying with CPU offload...")
            torch.cuda.empty_cache()
            # Fallback: enable sequential CPU offload if not already enabled
            pipe.enable_sequential_cpu_offload()
            generator = torch.Generator(device=DEVICE).manual_seed(GENERATION_SEED + i)
            image = pipe(prompt, num_inference_steps=15, height=512, width=512, generator=generator).images[0]
            output_filename = os.path.join(OUTPUT_DIR, f"output_frame_{i:05d}.png")
            image.save(output_filename)
        
    print(f"[+] Successfully generated {len(captions)} new images in '{OUTPUT_DIR}'.")

def create_video_with_audio(output_path, fps, num_frames, has_audio=False, original_video_path=None, pip_enabled=False, pip_size="1/4", pip_position="top-right"):
    """
    Compiles a sequence of images into a video file, optionally with audio and picture-in-picture.
    
    Args:
        output_path (str): The path for the final output video.
        fps (int): The frames per second for the output video.
        num_frames (int): The number of frames to include in the video.
        has_audio (bool): Whether to include audio from the original video.
        original_video_path (str): Path to original video for PiP overlay.
        pip_enabled (bool): Whether to add picture-in-picture overlay.
        pip_size (str): Size of PiP overlay.
        pip_position (str): Position of PiP overlay.
    """
    pip_text = " with picture-in-picture" if pip_enabled else ""
    print(f"[*] Step 4: Compiling {num_frames} frames into video '{output_path}' at {fps} FPS{pip_text}...")
    
    # Create a temporary video file without audio first
    temp_video = "temp_video_no_audio.mp4"
    
    try:
        # Method 1: Use ffmpeg for better quality and compatibility
        if _create_video_with_ffmpeg(temp_video, fps, num_frames):
            if has_audio and Path(TEMP_AUDIO_FILE).exists():
                # Merge video and audio using ffmpeg, optionally with PiP
                _merge_video_and_audio(temp_video, TEMP_AUDIO_FILE, output_path, original_video_path, pip_enabled, pip_size, pip_position)
            elif pip_enabled and original_video_path:
                # Add PiP without audio
                _add_picture_in_picture(temp_video, original_video_path, output_path, pip_size, pip_position)
            else:
                # Just rename the temp video to final output
                Path(temp_video).rename(output_path)
                print(f"[+] Video created without audio: {output_path}")
        else:
            # Fallback: Use OpenCV method (your original method)
            print("[!] ffmpeg failed, falling back to OpenCV method...")
            _create_video_with_opencv(output_path, fps, num_frames)
            if has_audio or pip_enabled:
                print("[!] Audio and PiP cannot be added with OpenCV fallback method.")
                
    finally:
        # Clean up temporary video file
        if Path(temp_video).exists():
            Path(temp_video).unlink()

def _add_picture_in_picture(video_path, original_video_path, output_path, pip_size="1/4", pip_position="top-right"):
    """
    Adds picture-in-picture overlay without audio.
    """
    try:
        filter_complex = _get_pip_filter(pip_size, pip_position)
        cmd = [
            'ffmpeg',
            '-i', video_path,          # Main video (AI-generated)
            '-i', original_video_path,  # Original video for PiP
            '-filter_complex', filter_complex,
            '-map', '[v]',             # Use the overlayed video
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[+] Successfully created video with picture-in-picture: {output_path}")
        else:
            print(f"[!] Failed to add picture-in-picture: {result.stderr}")
            Path(video_path).rename(output_path)
            
    except Exception as e:
        print(f"[!] Error adding picture-in-picture: {e}")
        Path(video_path).rename(output_path)

def _create_video_with_ffmpeg(output_path, fps, num_frames):
    """
    Creates video using ffmpeg for better quality and format support.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', f'{OUTPUT_DIR}/output_frame_%05d.png',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',  # Good quality
            '-preset', 'medium',
            '-y',  # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[+] Video created successfully with ffmpeg")
            return True
        else:
            print(f"[!] ffmpeg video creation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[!] Error with ffmpeg: {e}")
        return False

def _get_pip_filter(pip_size, pip_position):
    """
    Generate ffmpeg filter for picture-in-picture based on size and position.
    
    Args:
        pip_size (str): Size specification (e.g., "1/4", "320:240")
        pip_position (str): Position ("top-left", "top-right", "bottom-left", "bottom-right")
    
    Returns:
        str: ffmpeg filter string
    """
    # Handle size specification
    if ':' in pip_size:
        # Explicit pixel dimensions
        scale_filter = f'scale={pip_size}'
    else:
        # Fractional scaling
        scale_filter = f'scale=iw*{pip_size}:ih*{pip_size}'
    
    # Handle position
    margin = 10  # 10 pixel margin from edges
    position_map = {
        'top-left': f'{margin}:{margin}',
        'top-right': f'main_w-overlay_w-{margin}:{margin}',
        'bottom-left': f'{margin}:main_h-overlay_h-{margin}',
        'bottom-right': f'main_w-overlay_w-{margin}:main_h-overlay_h-{margin}'
    }
    
    overlay_pos = position_map.get(pip_position, position_map['top-right'])
    
    return f'[1:v]{scale_filter}[pip];[0:v][pip]overlay={overlay_pos}[v]'

def _merge_video_and_audio(video_path, audio_path, output_path, original_video_path=None, pip_enabled=False, pip_size="1/4", pip_position="top-right"):
    """
    Merges video and audio files using ffmpeg, optionally with picture-in-picture overlay.
    
    Args:
        video_path (str): Path to the AI-generated video
        audio_path (str): Path to the extracted audio
        output_path (str): Path for the final output
        original_video_path (str): Path to original video for PiP overlay
        pip_enabled (bool): Whether to add picture-in-picture overlay
        pip_size (str): Size of PiP overlay
        pip_position (str): Position of PiP overlay
    """
    try:
        if pip_enabled and original_video_path:
            # Create video with picture-in-picture overlay
            filter_complex = _get_pip_filter(pip_size, pip_position)
            cmd = [
                'ffmpeg',
                '-i', video_path,          # Main video (AI-generated)
                '-i', original_video_path,  # Original video for PiP
                '-i', audio_path,          # Audio track
                '-filter_complex', filter_complex,
                '-map', '[v]',             # Use the overlayed video
                '-map', '2:a',             # Use the audio from the third input
                '-c:a', 'aac',
                '-shortest',
                '-y',
                output_path
            ]
        else:
            # Standard merge without PiP
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',  # Copy video stream
                '-c:a', 'aac',   # Re-encode audio to AAC
                '-shortest',     # End when shortest stream ends
                '-y',           # Overwrite output file
                output_path
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            if pip_enabled:
                print(f"[+] Successfully created video with audio and picture-in-picture: {output_path}")
            else:
                print(f"[+] Successfully created video with audio: {output_path}")
        else:
            print(f"[!] Failed to merge audio/PiP: {result.stderr}")
            # Fallback: just copy the video without audio
            Path(video_path).rename(output_path)
            print(f"[+] Video created without audio: {output_path}")
            
    except Exception as e:
        print(f"[!] Error merging audio/PiP: {e}")
        Path(video_path).rename(output_path)

def _create_video_with_opencv(output_path, fps, num_frames):
    """
    Fallback method using OpenCV (original create_video function).
    """
    # Get the dimensions of the first output image
    first_image_path = os.path.join(OUTPUT_DIR, "output_frame_00000.png")
    sample_image = cv2.imread(first_image_path)
    height, width, layers = sample_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        frame_path = os.path.join(OUTPUT_DIR, f"output_frame_{i:05d}.png")
        img = cv2.imread(frame_path)
        video.write(img)

    video.release()
    print(f"[+] Successfully created output video: {output_path}")

def cleanup():
    """Removes temporary directories and files."""
    print("[*] Cleaning up temporary files...")
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    if os.path.exists(TEMP_AUDIO_FILE):
        os.remove(TEMP_AUDIO_FILE)
    print("[+] Cleanup complete.")

def main():
    global GENERATION_SEED
    
    parser = argparse.ArgumentParser(description="AI Video-to-Video Transformation Experiment (v2v.py)")
    parser.add_argument("-i", "--input", required=True, help="Path to the input video file (e.g., input.mp4)")
    parser.add_argument("-o", "--output", required=True, help="Path for the output video file (e.g., output.mp4)")
    parser.add_argument("-fps", "--fps", type=int, default=3, help="Frames per second to process and for the final video.")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio extraction and create video without audio")
    parser.add_argument("--seed", type=int, default=GENERATION_SEED, help=f"Seed for image generation (default: {GENERATION_SEED})")
    parser.add_argument("--pip", action="store_true", help="Add picture-in-picture overlay of original video in top-right corner")
    parser.add_argument("--pip-size", type=str, default="1/4", help="Size of PiP overlay as fraction (e.g., '1/4', '1/3') or pixels (e.g., '320:240')")
    parser.add_argument("--pip-position", type=str, default="top-right", choices=["top-left", "top-right", "bottom-left", "bottom-right"], help="Position of PiP overlay")
    
    args = parser.parse_args()
    
    # Update the global seed if provided
    GENERATION_SEED = args.seed

    try:
        # --- The main pipeline ---
        num_frames = extract_frames(args.input, args.fps)
        if num_frames == 0:
            print("[!] No frames were extracted. Check the input video path and integrity.")
            return

        # Extract audio (unless --no-audio flag is used)
        has_audio = False
        if not args.no_audio:
            has_audio = extract_audio(args.input)

        captions = caption_frames(num_frames)
        generate_images_from_captions(captions)
        
        # Create video with or without audio and picture-in-picture
        create_video_with_audio(
            args.output, 
            args.fps, 
            num_frames, 
            has_audio, 
            args.input if args.pip else None, 
            args.pip, 
            args.pip_size, 
            args.pip_position
        )

    except Exception as e:
        print(f"\n[!!!] An error occurred: {e}")
        print("[!] The process was interrupted.")
    finally:
        # --- Cleanup ---
        cleanup()

if __name__ == "__main__":
    main()
