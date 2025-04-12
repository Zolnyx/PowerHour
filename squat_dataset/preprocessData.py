import os
import ffmpeg
from tqdm import tqdm  # For progress bars

def preprocess_video(input_path, output_path, fps=12, target_width=720, target_height=1280):
    """
    Standardizes videos to:
    - Consistent FPS (default: 12)
    - Consistent resolution (default: 720x1280)
    - Maintains aspect ratio with padding
    - Preserves original quality
    """
    try:
        # Probe the input video
        probe = ffmpeg.probe(input_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        
        if not video_stream:
            raise ValueError("No video stream found")
            
        # Calculate padding to maintain aspect ratio
        in_width = int(video_stream['width'])
        in_height = int(video_stream['height'])
        aspect_ratio = in_width / in_height
        target_ar = target_width / target_height
        
        if aspect_ratio > target_ar:
            # Wider than target - add vertical padding
            new_height = int(target_width / aspect_ratio)
            pad_top = (target_height - new_height) // 2
            pad_bottom = target_height - new_height - pad_top
            filter_str = f"scale={target_width}:-1,pad={target_width}:{target_height}:0:{pad_top}:black"
        else:
            # Taller than target - add horizontal padding
            new_width = int(target_height * aspect_ratio)
            pad_left = (target_width - new_width) // 2
            pad_right = target_width - new_width - pad_left
            filter_str = f"scale=-1:{target_height},pad={target_width}:{target_height}:{pad_left}:0:black"
        
        # Process with ffmpeg
        (
            ffmpeg
            .input(input_path)
            .filter('fps', fps=fps, round='up')
            .filter('scale', target_width, -1)  # Maintain aspect ratio
            .filter('pad', target_width, target_height, -1, -1, 'black')  # Center-pad to target size
            .output(output_path, vcodec='libx264', crf=18, preset='fast')
            .overwrite_output()
            .run(quiet=True)
        )
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def batch_preprocess(input_dir="raw_videos", output_dir="processed", exercise="squats"):
    """Process all videos in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.mp4', '.mov', '.avi'))]
    
    print(f"Found {len(video_files)} videos to process")
    
    for filename in tqdm(video_files, desc="Processing videos"):
        input_path = os.path.join(input_dir, filename)
        output_filename = f"{len(os.listdir(output_dir)):03d}_{exercise}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        if preprocess_video(input_path, output_path):
            print(f"\nProcessed: {filename} → {output_filename}")
        else:
            print(f"\nFailed: {filename}")

if __name__ == "__main__":
    # Example usage
    batch_preprocess(
        input_dir="raw_videos",  # Source directory
        output_dir="processed",  # Where to save standardized videos
        exercise="squats"       # Will create 000_squats.mp4, 001_squats.mp4 etc.
    )