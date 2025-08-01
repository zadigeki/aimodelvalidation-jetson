#!/usr/bin/env python3
"""
Simple Camera Test Script
Tests laptop camera functionality with OpenCV
"""

import cv2
import os
from datetime import datetime
from pathlib import Path

def test_camera():
    """Test camera access and capture frames"""
    
    print("🎥 Testing laptop camera...")
    
    # Create output directory
    output_dir = Path("camera_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return False
    
    print("✅ Camera opened successfully!")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Camera resolution: {width}x{height}")
    print(f"📹 Camera FPS: {fps}")
    
    frames_captured = 0
    max_frames = 5
    
    print(f"\n🎯 Capturing {max_frames} test frames...")
    print("Press 'q' to quit early, or wait for automatic completion")
    
    try:
        while frames_captured < max_frames:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"test_frame_{frames_captured+1}_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)
            
            # Display frame (optional - comment out if running headless)
            cv2.imshow('Camera Test', frame)
            
            frames_captured += 1
            print(f"📸 Captured frame {frames_captured}: {filename}")
            
            # Wait for key press or timeout
            key = cv2.waitKey(1000) & 0xFF  # Wait 1 second
            if key == ord('q'):
                print("🛑 Stopping capture (user pressed 'q')")
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping capture (user pressed Ctrl+C)")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"\n✅ Camera test complete!")
    print(f"📁 {frames_captured} frames saved to: {output_dir.absolute()}")
    
    # List captured files
    if frames_captured > 0:
        print("\n📋 Captured files:")
        for file in sorted(output_dir.glob("test_frame_*.jpg")):
            size = file.stat().st_size / 1024  # Size in KB
            print(f"   📄 {file.name} ({size:.1f} KB)")
    
    return frames_captured > 0

def check_camera_availability():
    """Check if camera is available"""
    print("🔍 Checking camera availability...")
    
    # Try different camera indices
    for i in range(3):  # Check indices 0, 1, 2
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Camera found at index {i}")
            cap.release()
            return i
        cap.release()
    
    print("❌ No camera found")
    return None

if __name__ == "__main__":
    print("🎥 AI Model Validation - Camera Test")
    print("=" * 50)
    
    # Check camera availability first
    camera_index = check_camera_availability()
    
    if camera_index is not None:
        # Test camera
        success = test_camera()
        
        if success:
            print("\n🎉 Camera test successful!")
            print("💡 Your camera is working and ready for the AI model validation pipeline")
        else:
            print("\n❌ Camera test failed")
    else:
        print("\n❌ No camera detected")
        print("💡 Make sure:")
        print("   - Camera is connected")
        print("   - Camera permissions are granted")
        print("   - No other application is using the camera")