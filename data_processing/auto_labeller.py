import os
import cv2
import pandas as pd
from datetime import datetime

class SquatLabeler:
    def __init__(self, video_dir="processed/", output_file="demodata/labels.csv"):
        self.video_dir = video_dir
        self.output_file = output_file
        self.current_labels = []
        
        # Tag combinations (modify as needed)
        self.valid_tags = {
            'c': 'Correct',
            'k': 'Knees Forward',
            'h': 'Hips Incorrect', 
            'r': 'Back Arch',
            'x': 'Good Depth',
            'i': 'Invalid'
        }
        
        # Key bindings (now supports combinations)
        self.key_bindings = {
            ord('1'): 'c',
            ord('2'): 'k',
            ord('3'): 'h',
            ord('4'): 'r',
            ord('5'): 'x',
            ord('0'): 'i',
            ord(' '): 'toggle_pause',
            ord('s'): 'save',
            ord('q'): 'quit'
        }

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_id = os.path.basename(video_path).split('_')[0]
        current_tags = set()
        
        print(f"\nLabeling {os.path.basename(video_path)}")
        print("Keys: 1=c 2=k 3=h 4=r 5=x 0=i | Space=Toggle Tags | S=Save | Q=Quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = round(frame_num / fps, 2)
            
            # Display current tags
            tag_display = ''.join(sorted(current_tags)) if current_tags else 'i'
            cv2.putText(frame, f"Current: {tag_display}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show instructions
            y = 60
            for key, tag in self.valid_tags.items():
                cv2.putText(frame, f"{key}: {tag}", (20, y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y += 25
            
            cv2.imshow("Squat Labeler", frame)
            
            key = cv2.waitKey(0) & 0xFF
            action = self.key_bindings.get(key, None)
            
            if action == 'quit':
                break
            elif action == 'save':
                self._save_frame(video_id, timestamp, current_tags)
            elif action == 'toggle_pause':
                # Toggle tags for this frame
                if current_tags:
                    self._save_frame(video_id, timestamp, current_tags)
                    current_tags = set()
            elif action in self.valid_tags:
                # Toggle individual tags
                if action in current_tags:
                    current_tags.remove(action)
                else:
                    current_tags.add(action)
        
        cap.release()
        if current_tags:  # Save remaining tags
            self._save_frame(video_id, timestamp, current_tags)

    def _save_frame(self, video_id, timestamp, tags):
        if not tags:
            tags = {'i'}  # Default to invalid if empty
            
        label = ''.join(sorted(tags))  # Creates combinations like 'hx'
        self.current_labels.append({
            'video_id': video_id,
            'timestamp': timestamp,
            'label': label
        })
        print(f"Saved: {timestamp}s - {label}")

    def save_to_csv(self):
        if not self.current_labels:
            print("No labels to save!")
            return
            
        df = pd.DataFrame(self.current_labels)
        
        # Sort by video_id and timestamp
        df = df.sort_values(['video_id', 'timestamp'])
        
        # Save in your exact format (tab-separated, no header)
        df.to_csv(self.output_file, sep='\t', header=False, index=False, mode='a')
        print(f"\nSaved {len(df)} labels to {self.output_file}")
        self.current_labels = []  # Clear after saving

    def run(self):
        videos = sorted([f for f in os.listdir(self.video_dir) 
                       if f.endswith(('.mp4', '.avi', '.mov'))])
        
        for video in videos:
            self.process_video(os.path.join(self.video_dir, video))
            self.save_to_csv()
            
        cv2.destroyAllWindows()

if __name__ == "__main__":
    labeler = SquatLabeler(
        video_dir="processed",
        output_file="demodata/labels.csv"
    )
    labeler.run()