import os
import cv2
import pandas as pd
from datetime import datetime

class SquatLabeler:
    def __init__(self, video_dir="videos", output_file="labels.csv"):
        self.video_dir = video_dir
        self.output_file = output_file
        self.current_labels = []
        
        # Key mappings (same as before)
        self.label_keys = {
            49: 'c',  # 1
            50: 'k',  # 2
            51: 'h',  # 3
            52: 'r',  # 4
            53: 'x',  # 5
            48: 'i'   # 0
        }

    def process_video(self, video_path):
        video_name = os.path.basename(video_path)
        video_id = video_name.split('_')[0]  # Extract 000 from 000_squat.mp4
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\nLabeling {video_name} (Press Q to finish)")
        print("Keys: 1=c 2=k 3=h 4=r 5=x 0=i")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = round(frame_num / fps, 2)
            
            # Display frame info
            cv2.putText(frame, f"{video_name} | {timestamp}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Squat Labeler", frame)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key in self.label_keys:
                label = self.label_keys[key]
                self.current_labels.append({
                    'video_id': video_id,
                    'timestamp': timestamp,
                    'label': label
                })
                print(f"Added: {timestamp}s - {label}")
        
        cap.release()
        return True

    def save_labels(self):
        if not self.current_labels:
            print("No labels to save!")
            return
            
        # Convert to original format
        df = pd.DataFrame(self.current_labels)
        df = df[['video_id', 'timestamp', 'label']]  # Reorder columns
        
        # Save to CSV (append if file exists)
        header = not os.path.exists(self.output_file)
        df.to_csv(self.output_file, mode='a', header=header, index=False, sep='\t')
        print(f"Saved {len(df)} labels to {self.output_file}")

    def run(self):
        videos = sorted([f for f in os.listdir(self.video_dir) 
                        if f.endswith('.mp4')])
        
        for video in videos:
            self.process_video(os.path.join(self.video_dir, video))
        
        self.save_labels()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    labeler = SquatLabeler(
        video_dir="processed",
        output_file="labels.csv"  # Will match your existing format
    )
    labeler.run()