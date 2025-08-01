#!/usr/bin/env python3
"""
AI Model Validation PoC - Real Camera Demo
Uses your laptop's camera for the complete workflow
"""

import os
import sys
import time
import json
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

class DemoLogger:
    """Simple colored logging for demo"""
    
    @staticmethod
    def info(message):
        print(f"ℹ️  {message}")
    
    @staticmethod
    def success(message):
        print(f"✅ {message}")
    
    @staticmethod
    def warning(message):
        print(f"⚠️  {message}")
    
    @staticmethod
    def error(message):
        print(f"❌ {message}")
    
    @staticmethod
    def step(step_num, message):
        print(f"\n🎯 Step {step_num}: {message}")
        print("=" * 50)

class RealWebcamCaptureService:
    """Real webcam service using OpenCV"""
    
    def __init__(self, output_dir="demo_data/captured_images", camera_index=0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_index = camera_index
        self.cap = None
        self.is_active = False
        
    def initialize(self):
        """Initialize camera"""
        DemoLogger.info("Initializing real camera...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera at index {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.is_active = True
        DemoLogger.success("Camera initialized successfully!")
        
        # Get camera info
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        DemoLogger.info(f"Camera resolution: {width}x{height}")
        DemoLogger.info(f"Camera FPS: {fps}")
    
    def capture_frames(self, count=3):
        """Capture real frames from camera"""
        if not self.is_active:
            raise Exception("Camera not initialized. Call initialize() first.")
        
        captured_files = []
        
        DemoLogger.info(f"📸 Capturing {count} frames from your camera...")
        DemoLogger.info("Look at your camera - capturing in 3 seconds...")
        time.sleep(3)  # Give user time to prepare
        
        for i in range(count):
            DemoLogger.info(f"Capturing frame {i+1}/{count}...")
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                DemoLogger.error(f"Failed to capture frame {i+1}")
                continue
            
            # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = self.output_dir / f"real_frame_{i+1}_{timestamp}.jpg"
            
            cv2.imwrite(str(filename), frame)
            captured_files.append(filename)
            
            DemoLogger.success(f"📸 Frame {i+1} saved: {filename.name}")
            
            # Show preview (optional)
            cv2.imshow('Captured Frame', frame)
            cv2.waitKey(1000)  # Show for 1 second
            
            time.sleep(1)  # Wait between captures
        
        cv2.destroyAllWindows()
        return captured_files
    
    def cleanup(self):
        """Clean up camera resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.is_active = False
        DemoLogger.info("Camera resources cleaned up")

class MockAnnotationService:
    """Mock CVAT annotation service"""
    
    def __init__(self, output_dir="demo_data/annotations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_project(self, project_name):
        DemoLogger.info(f"🏗️  Creating CVAT project: {project_name}")
        time.sleep(1)
        return {"project_id": f"cvat_project_{int(time.time())}", "name": project_name}
    
    def upload_images(self, image_files):
        DemoLogger.info(f"📤 Uploading {len(image_files)} images to CVAT...")
        time.sleep(2)
        
        # Create mock annotations for each image
        annotations = []
        for img_file in image_files:
            annotation = {
                "image": str(img_file),
                "annotations": [
                    {
                        "label": "person",
                        "bbox": [100, 100, 200, 200],
                        "confidence": 0.95
                    },
                    {
                        "label": "object",
                        "bbox": [300, 150, 100, 100],
                        "confidence": 0.87
                    }
                ]
            }
            annotations.append(annotation)
        
        # Save annotations
        annotation_file = self.output_dir / "annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f, indent=2, default=str)
        
        DemoLogger.success(f"✅ Annotations saved: {annotation_file}")
        return annotation_file

class MockValidationService:
    """Mock Deepchecks validation service"""
    
    def __init__(self, output_dir="demo_data/validation_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_dataset(self, annotation_file):
        DemoLogger.info("🔍 Running Deepchecks data validation...")
        time.sleep(2)
        
        # Create mock validation report
        validation_report = {
            "validation_timestamp": datetime.now().isoformat(),
            "dataset_quality": {
                "overall_score": 0.92,
                "image_quality": 0.95,
                "annotation_quality": 0.89,
                "data_distribution": 0.91
            },
            "checks": [
                {"name": "Image Quality Check", "status": "PASSED", "score": 0.95},
                {"name": "Label Distribution", "status": "PASSED", "score": 0.91},
                {"name": "Annotation Consistency", "status": "WARNING", "score": 0.85},
                {"name": "Data Completeness", "status": "PASSED", "score": 0.98}
            ],
            "recommendations": [
                "Consider adding more diverse backgrounds",
                "Verify annotation consistency for 'object' class"
            ]
        }
        
        report_file = self.output_dir / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        DemoLogger.success(f"✅ Validation report: {report_file}")
        return validation_report

class MockModelTrainingService:
    """Mock Ultralytics YOLO training service"""
    
    def __init__(self, output_dir="demo_data/models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_model(self, annotation_file, validation_report):
        DemoLogger.info("🧠 Training YOLO model with your captured data...")
        
        # Simulate training progress
        epochs = 5
        for epoch in range(epochs):
            loss = 0.5 - (epoch * 0.08)  # Decreasing loss
            accuracy = 0.7 + (epoch * 0.05)  # Increasing accuracy
            
            DemoLogger.info(f"Epoch {epoch+1}/{epochs}: Loss={loss:.3f}, Accuracy={accuracy:.3f}")
            time.sleep(1)
        
        # Create mock model metadata
        model_metadata = {
            "training_timestamp": datetime.now().isoformat(),
            "model_type": "YOLOv8n",
            "epochs": epochs,
            "final_metrics": {
                "loss": 0.12,
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.86,
                "f1_score": 0.87
            },
            "training_data": str(annotation_file),
            "validation_score": validation_report["dataset_quality"]["overall_score"]
        }
        
        model_file = self.output_dir / "trained_model_metadata.json"
        with open(model_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        DemoLogger.success(f"✅ Model training complete: {model_file}")
        return model_metadata

class AIModelValidationDemo:
    """Main demo orchestrator with real camera"""
    
    def __init__(self):
        self.base_dir = Path("demo_data")
        self.base_dir.mkdir(exist_ok=True)
        
        # Initialize services
        self.webcam_service = RealWebcamCaptureService()
        self.annotation_service = MockAnnotationService()
        self.validation_service = MockValidationService()
        self.training_service = MockModelTrainingService()
    
    def run_complete_workflow(self, frame_count=3, project_name="Real Camera Demo"):
        """Run the complete AI model validation workflow"""
        
        try:
            DemoLogger.step(1, "Real Camera Data Capture")
            self.webcam_service.initialize()
            captured_files = self.webcam_service.capture_frames(frame_count)
            
            DemoLogger.step(2, "CVAT Annotation Workflow (Mock)")
            project = self.annotation_service.create_project(project_name)
            annotation_file = self.annotation_service.upload_images(captured_files)
            
            DemoLogger.step(3, "Deepchecks Data Validation (Mock)")
            validation_report = self.validation_service.validate_dataset(annotation_file)
            
            DemoLogger.step(4, "Ultralytics YOLO Training (Mock)")
            model_metadata = self.training_service.train_model(annotation_file, validation_report)
            
            DemoLogger.step(5, "Workflow Summary")
            self.print_summary(captured_files, validation_report, model_metadata)
            
        except Exception as e:
            DemoLogger.error(f"Demo failed: {e}")
            raise
        finally:
            self.webcam_service.cleanup()
    
    def print_summary(self, captured_files, validation_report, model_metadata):
        """Print workflow summary"""
        
        print("\n" + "="*60)
        print("🎉 AI MODEL VALIDATION WORKFLOW COMPLETE!")
        print("="*60)
        
        print(f"\n📸 DATA CAPTURE:")
        print(f"   • Captured {len(captured_files)} real images from your camera")
        for i, file in enumerate(captured_files, 1):
            size = file.stat().st_size / 1024
            print(f"   • Frame {i}: {file.name} ({size:.1f} KB)")
        
        print(f"\n✅ DATA VALIDATION:")
        score = validation_report["dataset_quality"]["overall_score"]
        print(f"   • Overall Quality Score: {score:.1%}")
        print(f"   • Image Quality: {validation_report['dataset_quality']['image_quality']:.1%}")
        print(f"   • Annotation Quality: {validation_report['dataset_quality']['annotation_quality']:.1%}")
        
        print(f"\n🧠 MODEL TRAINING:")
        metrics = model_metadata["final_metrics"]
        print(f"   • Model Type: {model_metadata['model_type']}")
        print(f"   • Final Accuracy: {metrics['accuracy']:.1%}")
        print(f"   • F1 Score: {metrics['f1_score']:.2f}")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"   • Images: {self.base_dir}/captured_images/")
        print(f"   • Annotations: {self.base_dir}/annotations/")
        print(f"   • Validation: {self.base_dir}/validation_reports/")
        print(f"   • Models: {self.base_dir}/models/")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   • Review captured images in the output directory")
        print(f"   • Check validation report for data quality insights")
        print(f"   • Model is ready for integration with real CVAT, Deepchecks, and YOLO")

def main():
    """Main demo function"""
    
    print("="*60)
    print("🤖 AI MODEL VALIDATION POC - REAL CAMERA DEMO")
    print("="*60)
    print("This demo uses your laptop's camera for real data capture!")
    print("• 📷 Real webcam data capture")
    print("• 🖼️  CVAT annotation workflow (mock)")
    print("• ✅ Deepchecks data validation (mock)")
    print("• 🧠 Ultralytics YOLO training (mock)")
    print("• 📊 Comprehensive reporting")
    print("="*60)
    
    try:
        # Run demo with default settings
        demo = AIModelValidationDemo()
        demo.run_complete_workflow(
            frame_count=3,
            project_name="Real Camera AI Validation"
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        DemoLogger.error(f"Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()