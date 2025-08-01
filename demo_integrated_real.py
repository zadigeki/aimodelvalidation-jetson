#!/usr/bin/env python3
"""
AI Model Validation PoC - Integrated Demo with Real Services
Uses your laptop's camera with real CVAT, Deepchecks, and Ultralytics integrations
"""

import os
import sys
import time
import json
import asyncio
import cv2
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from services.real_services import (
    RealServiceOrchestrator,
    DemoLogger
)

class RealWebcamCaptureService:
    """Real webcam service using OpenCV"""
    
    def __init__(self, output_dir="demo_data/real_captured", camera_index=0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_index = camera_index
        self.cap = None
        self.is_active = False
        
    def initialize(self):
        """Initialize camera"""
        DemoLogger.info("Initializing real camera for integrated workflow...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera at index {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.is_active = True
        DemoLogger.success("Camera initialized for real service integration!")
        
        # Get camera info
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        DemoLogger.info(f"Camera settings: {width}x{height} @ {fps}fps")
        return True
    
    def capture_frames(self, count=5):
        """Capture real frames from camera for real service processing"""
        if not self.is_active:
            raise Exception("Camera not initialized. Call initialize() first.")
        
        captured_files = []
        
        DemoLogger.info(f"📸 Capturing {count} frames for real AI workflow...")
        DemoLogger.info("Position yourself in front of the camera - starting in 3 seconds...")
        time.sleep(3)
        
        for i in range(count):
            DemoLogger.info(f"Capturing frame {i+1}/{count}...")
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                DemoLogger.error(f"Failed to capture frame {i+1}")
                continue
            
            # Save frame with descriptive naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = self.output_dir / f"real_ai_frame_{i+1:02d}_{timestamp}.jpg"
            
            cv2.imwrite(str(filename), frame)
            captured_files.append(filename)
            
            DemoLogger.success(f"📸 Frame {i+1} captured: {filename.name}")
            
            # Show preview
            cv2.imshow('AI Training Data Capture', frame)
            cv2.waitKey(1500)  # Show for 1.5 seconds
            
            if i < count - 1:  # Don't wait after last frame
                time.sleep(1)
        
        cv2.destroyAllWindows()
        return captured_files
    
    def cleanup(self):
        """Clean up camera resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.is_active = False
        DemoLogger.info("Camera resources cleaned up")

class IntegratedAIWorkflowDemo:
    """Complete AI workflow with real camera and real services"""
    
    def __init__(self):
        self.base_dir = Path("demo_data")
        self.base_dir.mkdir(exist_ok=True)
        
        # Initialize services
        self.webcam_service = RealWebcamCaptureService()
        self.orchestrator = RealServiceOrchestrator()
    
    async def run_complete_integrated_workflow(self, frame_count=5, project_name="Integrated Real AI Validation"):
        """Run the complete integrated AI workflow"""
        
        workflow_start = datetime.now()
        DemoLogger.info(f"🚀 Starting integrated AI workflow: {project_name}")
        
        try:
            # Phase 1: Real Camera Data Capture
            DemoLogger.step(1, "Real Camera Data Capture")
            DemoLogger.info("Using your laptop camera to capture training data...")
            
            self.webcam_service.initialize()
            captured_files = self.webcam_service.capture_frames(frame_count)
            
            DemoLogger.success(f"✅ Captured {len(captured_files)} images for AI training")
            
            # Phase 2: Complete Real Services Workflow
            DemoLogger.step(2, "Real AI Services Integration")
            DemoLogger.info("Processing your images through real AI pipeline...")
            
            workflow_results = await self.orchestrator.run_complete_workflow(
                captured_files, 
                project_name
            )
            
            # Phase 3: Results Analysis and Summary
            DemoLogger.step(3, "Workflow Results Analysis")
            self.analyze_and_display_results(captured_files, workflow_results, workflow_start)
            
            return {
                "captured_files": captured_files,
                "workflow_results": workflow_results,
                "duration": (datetime.now() - workflow_start).total_seconds()
            }
            
        except Exception as e:
            DemoLogger.error(f"Integrated workflow failed: {e}")
            raise
        finally:
            self.webcam_service.cleanup()
    
    def analyze_and_display_results(self, captured_files, workflow_results, start_time):
        """Analyze and display comprehensive workflow results"""
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("🎉 INTEGRATED AI MODEL VALIDATION COMPLETE!")
        print("="*80)
        
        print(f"\n⏱️  WORKFLOW SUMMARY:")
        print(f"   • Total Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"   • Project Name: {workflow_results.get('project_name', 'Unknown')}")
        print(f"   • Workflow ID: {workflow_results.get('workflow_id', 'Unknown')}")
        print(f"   • Status: {workflow_results.get('status', 'Unknown').upper()}")
        
        print(f"\n📸 REAL CAMERA DATA CAPTURE:")
        print(f"   • Images Captured: {len(captured_files)}")
        total_size = sum(f.stat().st_size for f in captured_files) / 1024
        print(f"   • Total Size: {total_size:.1f} KB")
        print(f"   • Average Size: {total_size/len(captured_files):.1f} KB per image")
        
        # Display individual files
        for i, file in enumerate(captured_files, 1):
            size = file.stat().st_size / 1024
            print(f"      📄 Frame {i}: {file.name} ({size:.1f} KB)")
        
        results = workflow_results.get("results", {})
        
        # CVAT Integration Results
        if "cvat_project" in results:
            cvat_data = results["cvat_project"]
            print(f"\n🖼️  CVAT ANNOTATION INTEGRATION:")
            print(f"   • Project ID: {cvat_data.get('id', 'Unknown')}")
            print(f"   • Labels Defined: {len(cvat_data.get('labels', []))}")
            print(f"   • Status: {cvat_data.get('status', 'Unknown').upper()}")
            
            if "annotations" in results:
                ann_data = results["annotations"]
                print(f"   • Total Annotations: {ann_data.get('annotation_count', 0)}")
                print(f"   • Annotation File: {Path(str(ann_data.get('annotation_file', ''))).name}")
        
        # Deepchecks Validation Results
        if "validation" in results:
            val_data = results["validation"]
            quality = val_data.get("dataset_quality", {})
            
            print(f"\n✅ DEEPCHECKS DATA VALIDATION:")
            print(f"   • Overall Quality Score: {quality.get('overall_score', 0):.1%}")
            print(f"   • Image Quality: {quality.get('image_quality', 0):.1%}")
            print(f"   • Annotation Quality: {quality.get('annotation_quality', 0):.1%}")
            print(f"   • Data Distribution: {quality.get('data_distribution', 0):.1%}")
            
            checks = val_data.get("checks", [])
            passed_checks = sum(1 for check in checks if check.get("status") == "PASSED")
            print(f"   • Validation Checks: {passed_checks}/{len(checks)} PASSED")
            
            recommendations = val_data.get("recommendations", [])
            if recommendations:
                print(f"   • Key Recommendations:")
                for rec in recommendations[:3]:  # Show top 3
                    print(f"      • {rec}")
        
        # Ultralytics Training Results
        if "training" in results:
            train_data = results["training"]
            metrics = train_data.get("final_metrics", {})
            summary = train_data.get("training_summary", {})
            
            print(f"\n🧠 YOLO MODEL TRAINING:")
            print(f"   • Model Type: {summary.get('model_type', 'Unknown')}")
            print(f"   • Epochs Completed: {summary.get('epochs_completed', 0)}")
            print(f"   • Training Time: {summary.get('training_time_minutes', 0):.1f} minutes")
            print(f"   • Model Size: {summary.get('model_size_mb', 0):.1f} MB")
            print(f"   • Parameters: {summary.get('parameters', 0):,}")
            
            print(f"\n📊 MODEL PERFORMANCE METRICS:")
            print(f"   • mAP@50: {metrics.get('mAP50', 0):.3f}")
            print(f"   • mAP@50-95: {metrics.get('mAP50-95', 0):.3f}")
            print(f"   • Precision: {metrics.get('precision', 0):.3f}")
            print(f"   • Recall: {metrics.get('recall', 0):.3f}")
            print(f"   • F1-Score: {metrics.get('f1_score', 0):.3f}")
        
        # Model Evaluation Results
        if "evaluation" in results:
            eval_data = results["evaluation"]
            eval_metrics = eval_data.get("metrics", {})
            
            print(f"\n📈 MODEL EVALUATION:")
            print(f"   • Final mAP@50: {eval_metrics.get('mAP50', 0):.3f}")
            print(f"   • Final Precision: {eval_metrics.get('precision', 0):.3f}")
            print(f"   • Final Recall: {eval_metrics.get('recall', 0):.3f}")
            
            speed = eval_data.get("inference_speed", {})
            if speed:
                print(f"   • Inference Speed: {speed.get('total_ms', 0):.1f}ms per image")
        
        # File Outputs
        print(f"\n📁 GENERATED FILES AND OUTPUTS:")
        print(f"   • Original Images: demo_data/real_captured/")
        print(f"   • CVAT Annotations: demo_data/real_annotations/")
        print(f"   • Validation Reports: demo_data/real_validation/")
        print(f"   • YOLO Dataset: demo_data/real_models/yolo_dataset/")
        print(f"   • Trained Models: demo_data/real_models/")
        
        # Success Assessment
        overall_success = self.assess_workflow_success(workflow_results)
        
        print(f"\n🎯 WORKFLOW ASSESSMENT:")
        print(f"   • Overall Success: {'✅ EXCELLENT' if overall_success >= 0.9 else '✅ GOOD' if overall_success >= 0.8 else '⚠️ NEEDS IMPROVEMENT'}")
        print(f"   • Success Score: {overall_success:.1%}")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   • Review all generated files in demo_data/ directory")
        print(f"   • Examine validation reports for data quality insights")
        print(f"   • Test model predictions on new images")
        print(f"   • Consider training with more data for production use")
        print(f"   • Deploy model for real-world applications")
        
        print(f"\n🚀 INTEGRATION SUCCESS!")
        print(f"Your real camera data has been processed through the complete")
        print(f"AI model validation pipeline using real industry-standard tools!")
    
    def assess_workflow_success(self, workflow_results):
        """Assess overall workflow success score"""
        
        if workflow_results.get("status") != "completed":
            return 0.5
        
        results = workflow_results.get("results", {})
        scores = []
        
        # Data capture success
        if "annotations" in results:
            scores.append(1.0)  # Successfully captured and annotated
        
        # Validation success
        if "validation" in results:
            val_quality = results["validation"].get("dataset_quality", {})
            overall_score = val_quality.get("overall_score", 0.8)
            scores.append(overall_score)
        
        # Training success
        if "training" in results:
            train_metrics = results["training"].get("final_metrics", {})
            map50 = train_metrics.get("mAP50", 0.8)
            scores.append(map50)
        
        # Evaluation success
        if "evaluation" in results:
            eval_metrics = results["evaluation"].get("metrics", {})
            eval_map50 = eval_metrics.get("mAP50", 0.8)
            scores.append(eval_map50)
        
        return sum(scores) / len(scores) if scores else 0.7

def print_welcome_banner():
    """Print welcome banner for integrated demo"""
    print("="*80)
    print("🤖 AI MODEL VALIDATION - INTEGRATED REAL SERVICES DEMO")
    print("="*80)
    print("This demo provides a complete end-to-end AI model validation workflow:")
    print("")
    print("📷 REAL CAMERA CAPTURE:")
    print("   • Uses your laptop's camera to capture training images")
    print("   • Captures high-quality frames for model training")
    print("")
    print("🔗 REAL SERVICE INTEGRATIONS:")
    print("   • 🖼️  CVAT: Computer Vision Annotation Tool integration")
    print("   • ✅ Deepchecks: Comprehensive data validation and quality assessment")
    print("   • 🧠 Ultralytics YOLO: State-of-the-art object detection training")
    print("")
    print("🎯 COMPLETE WORKFLOW:")
    print("   • Data capture → Annotation → Validation → Training → Evaluation")
    print("   • Real industry-standard tools and frameworks")
    print("   • Production-ready AI model validation pipeline")
    print("")
    print("⚡ ADVANCED FEATURES:")
    print("   • Automatic COCO format annotation generation")
    print("   • Multi-dimensional data quality assessment")
    print("   • YOLO model training with real performance metrics")
    print("   • Comprehensive workflow result analysis")
    print("="*80)

async def main():
    """Main demo function"""
    
    print_welcome_banner()
    
    try:
        # Create integrated demo
        demo = IntegratedAIWorkflowDemo()
        
        # Run complete workflow
        DemoLogger.info("🚀 Starting integrated real services workflow...")
        
        results = await demo.run_complete_integrated_workflow(
            frame_count=5,
            project_name="Laptop Camera Real AI Training"
        )
        
        DemoLogger.success("🎉 Integrated workflow completed successfully!")
        
        return results
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        DemoLogger.error(f"Demo failed: {e}")
        print(f"\n💡 Troubleshooting tips:")
        print(f"   • Make sure camera permissions are granted")
        print(f"   • Check that all dependencies are installed")
        print(f"   • Ensure sufficient disk space for model training")
        print(f"   • Review error logs for specific issues")
        sys.exit(1)

if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main())