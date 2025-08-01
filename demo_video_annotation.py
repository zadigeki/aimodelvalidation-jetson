#!/usr/bin/env python3
"""
Demo script showing how video annotation works
Creates a sample annotated frame to demonstrate the output
"""

import cv2
import numpy as np
from pathlib import Path

def create_demo_annotation():
    """Create a demo frame showing annotation style"""
    # Create a sample frame (simulating video frame)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame.fill(50)  # Dark gray background
    
    # Add some simulated objects (rectangles to represent cars, trucks, etc.)
    # Car 1 - High confidence (green)
    cv2.rectangle(frame, (100, 200), (250, 300), (100, 100, 100), -1)  # Object shape
    cv2.rectangle(frame, (100, 200), (250, 300), (0, 255, 0), 2)      # Green bounding box
    cv2.rectangle(frame, (100, 175), (180, 200), (0, 255, 0), -1)     # Label background
    cv2.putText(frame, "car: 0.89", (105, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Truck - Medium confidence (yellow)
    cv2.rectangle(frame, (300, 180), (450, 320), (80, 80, 80), -1)    # Object shape
    cv2.rectangle(frame, (300, 180), (450, 320), (0, 255, 255), 2)    # Yellow bounding box
    cv2.rectangle(frame, (300, 155), (400, 180), (0, 255, 255), -1)   # Label background
    cv2.putText(frame, "truck: 0.67", (305, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Bus - Low confidence (red)
    cv2.rectangle(frame, (480, 220), (580, 350), (60, 60, 60), -1)    # Object shape
    cv2.rectangle(frame, (480, 220), (580, 350), (0, 0, 255), 2)      # Red bounding box
    cv2.rectangle(frame, (480, 195), (565, 220), (0, 0, 255), -1)     # Label background
    cv2.putText(frame, "bus: 0.43", (485, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add frame info overlay (bottom of frame)
    frame_info = "Frame 240 | 00:08.00 | Objects: 3"
    cv2.putText(frame, frame_info, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, frame_info, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Add legend
    legend_y = 30
    cv2.putText(frame, "Confidence Legend:", (450, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(frame, (450, legend_y + 10), (470, legend_y + 25), (0, 255, 0), -1)
    cv2.putText(frame, "High (>=0.8)", (480, legend_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(frame, (450, legend_y + 30), (470, legend_y + 45), (0, 255, 255), -1)
    cv2.putText(frame, "Medium (0.5-0.8)", (480, legend_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(frame, (450, legend_y + 50), (470, legend_y + 65), (0, 0, 255), -1)
    cv2.putText(frame, "Low (<0.5)", (480, legend_y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame

def main():
    print("🎬 Creating demo annotation frame...")
    
    # Create demo frame
    demo_frame = create_demo_annotation()
    
    # Save demo frame
    output_dir = Path("demo_data/annotation_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "sample_annotated_frame.jpg"
    cv2.imwrite(str(output_path), demo_frame)
    
    print(f"✅ Demo annotation saved: {output_path}")
    print()
    print("📋 Annotation Features:")
    print("  🟢 Green boxes: High confidence detections (≥80%)")
    print("  🟡 Yellow boxes: Medium confidence detections (50-80%)")
    print("  🔴 Red boxes: Low confidence detections (<50%)")
    print("  📍 Frame info: Shows frame number, timestamp, object count")
    print("  🏷️  Labels: Object class + confidence percentage")
    print()
    print("🎯 This is what your annotated video will look like!")
    print("   Each frame with detections will have bounding boxes and labels")
    print("   You can cross-reference with the markdown report timestamps")

if __name__ == "__main__":
    main()