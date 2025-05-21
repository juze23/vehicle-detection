import cv2
from svm_pipeline import vehicle_detection_svm
import numpy as np

def process_video(input_path, output_path):
    # Open the video
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Processing video...")
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create a copy for lane detection visualization
        img_lane_augmented = np.copy(frame_rgb)
        # Dummy lane info
        lane_info = {
            'left_curverad': 0,
            'right_curverad': 0,
            'center_dist': 0,
            'curve_direction': 'straight',
            'curvature': 0,
            'dev_dir': 'center',
            'offset': 0.0
        }
        
        # Process the frame
        processed_frame = vehicle_detection_svm(frame_rgb, img_lane_augmented, lane_info)
        
        # Convert back to BGR for saving
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        
        # Write the frame
        out.write(processed_frame_bgr)
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print progress every 30 frames
            print(f"Processed {frame_count} frames...")
    
    # Release everything
    cap.release()
    out.release()
    print("Done! Output saved to:", output_path)

if __name__ == "__main__":
    # You can change these paths to your video files
    input_video = "examples/project_video.mp4"
    output_video = "examples/output_video.mp4"
    
    process_video(input_video, output_video) 