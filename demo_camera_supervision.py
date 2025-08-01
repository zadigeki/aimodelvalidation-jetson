#!/usr/bin/env python3
"""
Real Camera + Roboflow Supervision Integration Demo

This demo shows real-time object detection on your laptop camera feed
using the Roboflow Supervision library with YOLO models.
"""

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import time
from datetime import datetime
import os

def create_output_dir():
    """Create output directory for saving results"""
    output_dir = "demo_data/supervision_camera"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def main():
    print("\n" + "="*70)
    print("🎥 REAL-TIME CAMERA + SUPERVISION INTEGRATION DEMO")
    print("="*70)
    print("📷 Using your laptop camera for live object detection")
    print("🤖 Powered by YOLO + Roboflow Supervision")
    print("🎯 Press 'q' to quit, 's' to save screenshot")
    print("="*70 + "\n")
    
    # Create output directory
    output_dir = create_output_dir()
    
    # Initialize camera
    print("🔍 Initializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("✅ Camera initialized successfully!")
    
    # Load YOLO model
    print("\n🤖 Loading YOLO model...")
    model = YOLO('yolov8n.pt')  # Using nano model for speed
    print("✅ Model loaded!")
    
    # Initialize annotators
    box_annotator = sv.BoxAnnotator(
        thickness=2
    )
    
    # Statistics tracking
    frame_count = 0
    total_detections = 0
    start_time = time.time()
    
    print("\n🚀 Starting real-time detection...")
    print("="*70 + "\n")
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            frame_count += 1
            
            # Run detection
            results = model(frame)[0]
            
            # Convert to supervision format
            detections = sv.Detections.from_ultralytics(results)
            
            # Count detections
            num_detections = len(detections)
            total_detections += num_detections
            
            # Annotate frame
            labels = [
                f"{model.names[class_id]} {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
            
            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )
            
            # Add statistics overlay
            fps = frame_count / (time.time() - start_time)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Objects: {num_detections}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Total Detected: {total_detections}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('AI Model Validation - Supervision Demo', annotated_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{output_dir}/supervision_capture_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Screenshot saved: {filename}")
                
                # Save detection info
                info_file = f"{output_dir}/supervision_capture_{timestamp}.txt"
                with open(info_file, 'w') as f:
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Objects detected: {num_detections}\n")
                    f.write(f"Classes: {labels}\n")
                    f.write(f"FPS: {fps:.1f}\n")
                print(f"📝 Detection info saved: {info_file}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Detection interrupted by user")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        duration = time.time() - start_time
        avg_fps = frame_count / duration if duration > 0 else 0
        
        print("\n" + "="*70)
        print("📊 SESSION SUMMARY")
        print("="*70)
        print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"📹 Frames processed: {frame_count}")
        print(f"🎯 Total objects detected: {total_detections}")
        print(f"⚡ Average FPS: {avg_fps:.1f}")
        print(f"📁 Results saved to: {output_dir}/")
        print("="*70)
        
        print("\n✅ Demo complete! Thank you for testing Supervision integration!")

if __name__ == "__main__":
    main()