#!/usr/bin/env python3
"""
AI Model Validation PoC - Interactive Demo
Demonstrates the complete workflow with simulated camera and mock services
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("⚠️  OpenCV not available - using simulated camera")

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

class MockWebcamCaptureService:
    """Mock webcam service that simulates camera capture"""
    
    def __init__(self):
        self.is_active = False
        self.frame_count = 0
        self.demo_images_dir = Path("demo_data/captured_images")
        self.demo_images_dir.mkdir(parents=True, exist_ok=True)
    
    def initialize(self):
        """Initialize camera service"""
        DemoLogger.info("Initializing webcam capture service...")
        time.sleep(1)
        DemoLogger.success("Webcam service initialized")
        return True
    
    def start_stream(self):
        """Start camera stream"""
        DemoLogger.info("Starting camera stream...")
        self.is_active = True
        time.sleep(1)
        DemoLogger.success("Camera stream active")
        return True
    
    def capture_frame(self):
        """Capture a frame (simulated)"""
        if not self.is_active:
            raise Exception("Camera not active. Call start_stream() first.")
        
        self.frame_count += 1
        
        if OPENCV_AVAILABLE:
            # Create a simple test image with OpenCV
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add some visual elements
            cv2.rectangle(img, (50, 50), (590, 430), (0, 255, 0), 2)
            cv2.putText(img, f'Demo Frame {self.frame_count}', (60, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f'Timestamp: {datetime.now().strftime("%H:%M:%S")}', 
                       (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(img, 'AI Model Validation PoC', (60, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Add some mock objects to detect
            cv2.circle(img, (200, 300), 30, (255, 0, 0), -1)  # Blue circle
            cv2.rectangle(img, (400, 280), (500, 380), (0, 0, 255), -1)  # Red rectangle
            cv2.putText(img, 'Objects for Detection', (60, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Save frame
            filename = self.demo_images_dir / f"frame_{self.frame_count:03d}.jpg"
            cv2.imwrite(str(filename), img)
        else:
            # Create mock frame data without OpenCV
            filename = self.demo_images_dir / f"frame_{self.frame_count:03d}.txt"
            with open(filename, 'w') as f:
                f.write(f"Mock frame {self.frame_count} captured at {datetime.now()}")
        
        frame_data = {
            'id': f'frame-{self.frame_count:03d}',
            'timestamp': time.time(),
            'filename': str(filename),
            'metadata': {
                'width': 640,
                'height': 480,
                'format': 'jpeg' if OPENCV_AVAILABLE else 'text',
                'objects_detected': ['circle', 'rectangle'] if self.frame_count % 2 == 0 else ['rectangle']
            }
        }
        
        DemoLogger.info(f"📸 Captured frame {self.frame_count}: {filename}")
        time.sleep(0.5)  # Simulate capture time
        return frame_data
    
    def stop_stream(self):
        """Stop camera stream"""
        self.is_active = False
        DemoLogger.info("Camera stream stopped")
        return True

class MockCVATAnnotationService:
    """Mock CVAT service that simulates annotation workflow"""
    
    def __init__(self):
        self.project_id = None
        self.annotations_dir = Path("demo_data/annotations")
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
    
    def create_project(self, project_config):
        """Create annotation project"""
        project_name = project_config.get('name', 'demo-project')
        self.project_id = f"cvat-{int(time.time())}"
        
        DemoLogger.info(f"Creating CVAT project: {project_name}")
        time.sleep(2)
        DemoLogger.success(f"CVAT project created with ID: {self.project_id}")
        
        return {'project_id': self.project_id, 'name': project_name}
    
    def upload_data(self, image_paths=None):
        """Upload images to CVAT"""
        DemoLogger.info("Uploading captured images to CVAT...")
        time.sleep(2)
        
        # Create mock annotations
        annotations = {
            'project_id': self.project_id,
            'annotations': [
                {
                    'image_id': 'frame-001',
                    'objects': [
                        {'class': 'circle', 'bbox': [170, 270, 60, 60], 'confidence': 0.95},
                        {'class': 'rectangle', 'bbox': [400, 280, 100, 100], 'confidence': 0.92}
                    ]
                },
                {
                    'image_id': 'frame-002', 
                    'objects': [
                        {'class': 'rectangle', 'bbox': [400, 280, 100, 100], 'confidence': 0.89}
                    ]
                }
            ]
        }
        
        annotation_file = self.annotations_dir / f"annotations_{self.project_id}.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        DemoLogger.success(f"✏️  Annotations created: {len(annotations['annotations'])} images annotated")
        DemoLogger.info(f"Annotation file: {annotation_file}")
        return annotation_file

class MockDeepChecksValidationService:
    """Mock Deepchecks service that simulates data validation"""
    
    def __init__(self):
        self.reports_dir = Path("demo_data/validation_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_dataset(self, dataset_path=None):
        """Validate dataset quality"""
        DemoLogger.info("Running Deepchecks data validation...")
        time.sleep(3)
        
        # Simulate validation checks
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': 0.87,
            'checks': {
                'data_integrity': {'score': 0.95, 'status': 'PASS', 'issues': 0},
                'label_quality': {'score': 0.89, 'status': 'PASS', 'issues': 1},
                'data_drift': {'score': 0.92, 'status': 'PASS', 'issues': 0},
                'feature_distribution': {'score': 0.78, 'status': 'WARNING', 'issues': 2},
                'outlier_detection': {'score': 0.91, 'status': 'PASS', 'issues': 0}
            },
            'recommendations': [
                'Consider balancing class distribution',
                'Review feature scaling for better performance'
            ]
        }
        
        DemoLogger.success(f"🔍 Dataset validation completed - Overall Score: {validation_results['overall_score']:.2f}")
        
        for check_name, result in validation_results['checks'].items():
            status_emoji = "✅" if result['status'] == 'PASS' else "⚠️" if result['status'] == 'WARNING' else "❌"
            DemoLogger.info(f"  {status_emoji} {check_name}: {result['score']:.2f} ({result['issues']} issues)")
        
        return validation_results
    
    def generate_report(self, validation_results=None):
        """Generate validation report"""
        DemoLogger.info("Generating Deepchecks validation report...")
        time.sleep(1)
        
        report_file = self.reports_dir / f"validation_report_{int(time.time())}.json"
        
        report_data = validation_results or {
            'summary': 'Dataset validation completed successfully',
            'timestamp': datetime.now().isoformat(),
            'overall_quality': 'GOOD'
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        DemoLogger.success(f"📊 Validation report generated: {report_file}")
        return str(report_file)

class MockUltralyticsTrainingService:
    """Mock Ultralytics service that simulates YOLO training"""
    
    def __init__(self):
        self.model_dir = Path("demo_data/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.training_logs = []
    
    def initialize(self):
        """Initialize training environment"""
        DemoLogger.info("Initializing Ultralytics YOLO training environment...")
        time.sleep(1)
        DemoLogger.success("YOLO training environment ready")
        return True
    
    def train(self, training_config):
        """Train YOLO model"""
        epochs = training_config.get('epochs', 50)
        batch_size = training_config.get('batchSize', 16)
        
        DemoLogger.info(f"Starting YOLO training - Epochs: {epochs}, Batch Size: {batch_size}")
        
        # Simulate training epochs
        for epoch in range(1, min(epochs + 1, 6)):  # Show first 5 epochs for demo
            time.sleep(1)
            
            # Simulate training metrics
            train_loss = 0.5 - (epoch * 0.05) + np.random.normal(0, 0.02)
            val_loss = 0.6 - (epoch * 0.04) + np.random.normal(0, 0.03)
            mAP50 = 0.3 + (epoch * 0.1) + np.random.normal(0, 0.02)
            
            metrics = {
                'epoch': epoch,
                'train_loss': max(0.1, train_loss),
                'val_loss': max(0.1, val_loss),
                'mAP50': min(0.95, max(0.2, mAP50))
            }
            
            self.training_logs.append(metrics)
            
            DemoLogger.info(f"  Epoch {epoch}/{epochs}: "
                          f"Loss={metrics['train_loss']:.3f}, "
                          f"Val Loss={metrics['val_loss']:.3f}, "
                          f"mAP50={metrics['mAP50']:.3f}")
        
        if epochs > 5:
            DemoLogger.info(f"  ... (continuing for {epochs - 5} more epochs)")
            # Simulate final metrics
            final_metrics = {
                'epoch': epochs,
                'train_loss': 0.15,
                'val_loss': 0.18,
                'mAP50': 0.78
            }
            self.training_logs.append(final_metrics)
            DemoLogger.info(f"  Final: Loss={final_metrics['train_loss']:.3f}, "
                          f"mAP50={final_metrics['mAP50']:.3f}")
        
        # Save mock model
        model_file = self.model_dir / f"best_model_{int(time.time())}.pt"
        model_data = {
            'model_type': 'YOLOv8',
            'training_config': training_config,
            'final_metrics': self.training_logs[-1],
            'classes': ['circle', 'rectangle']
        }
        
        with open(str(model_file).replace('.pt', '.json'), 'w') as f:
            json.dump(model_data, f, indent=2)
        
        DemoLogger.success(f"🤖 Model training completed! Saved to: {model_file}")
        DemoLogger.info(f"Final mAP50: {self.training_logs[-1]['mAP50']:.3f}")
        
        return {
            'model_path': str(model_file),
            'training_logs': self.training_logs,
            'final_metrics': self.training_logs[-1]
        }

class AIModelValidationDemo:
    """Main demo orchestrator"""
    
    def __init__(self):
        self.webcam_service = MockWebcamCaptureService()
        self.annotation_service = MockCVATAnnotationService()
        self.validation_service = MockDeepChecksValidationService()
        self.training_service = MockUltralyticsTrainingService()
        
        # Create demo data directory
        self.demo_data_dir = Path("demo_data")
        self.demo_data_dir.mkdir(exist_ok=True)
    
    def display_banner(self):
        """Display demo banner"""
        print("\n" + "="*60)
        print("🤖 AI MODEL VALIDATION POC - INTERACTIVE DEMO")
        print("="*60)
        print("This demo showcases the complete AI model validation workflow:")
        print("• 📷 Webcam data capture (simulated)")
        print("• 🖼️  CVAT annotation workflow (mock)")
        print("• ✅ Deepchecks data validation (mock)")
        print("• 🧠 Ultralytics YOLO training (mock)")
        print("• 📊 Comprehensive reporting")
        print("="*60)
        
        if not OPENCV_AVAILABLE:
            DemoLogger.warning("OpenCV not available - using text-based simulation")
        else:
            DemoLogger.info("OpenCV available - generating demo images")
        
        print()
    
    def get_user_input(self):
        """Get workflow configuration from user"""
        print("🎛️  Configure your AI model validation workflow:")
        print()
        
        try:
            frame_count = int(input("📸 Number of frames to capture (1-10) [default: 3]: ") or "3")
            frame_count = max(1, min(10, frame_count))
        except ValueError:
            frame_count = 3
        
        try:
            epochs = int(input("🧠 Training epochs (1-100) [default: 10]: ") or "10")
            epochs = max(1, min(100, epochs))
        except ValueError:
            epochs = 10
        
        project_name = input("📝 Project name [default: 'demo-validation']: ") or "demo-validation"
        
        return {
            'project_name': project_name,
            'frame_count': frame_count,
            'epochs': epochs,
            'batch_size': 16
        }
    
    async def run_complete_workflow(self, config):
        """Execute the complete AI model validation workflow"""
        
        DemoLogger.step(1, "Data Capture Phase")
        
        # Initialize and start webcam
        self.webcam_service.initialize()
        self.webcam_service.start_stream()
        
        # Capture frames
        captured_frames = []
        for i in range(config['frame_count']):
            frame = self.webcam_service.capture_frame()
            captured_frames.append(frame)
        
        self.webcam_service.stop_stream()
        DemoLogger.success(f"✅ Captured {len(captured_frames)} frames")
        
        # --------------------------------------------------------
        DemoLogger.step(2, "Annotation Phase (CVAT)")
        
        # Create CVAT project and upload data
        project = self.annotation_service.create_project({'name': config['project_name']})
        annotation_file = self.annotation_service.upload_data()
        
        # --------------------------------------------------------
        DemoLogger.step(3, "Data Validation Phase (Deepchecks)")
        
        # Validate dataset
        validation_results = self.validation_service.validate_dataset()
        report_file = self.validation_service.generate_report(validation_results)
        
        # Check if validation passed
        validation_passed = validation_results['overall_score'] >= 0.7
        
        if not validation_passed:
            DemoLogger.error("❌ Dataset validation failed! Cannot proceed to training.")
            return {
                'success': False,
                'error': 'Dataset validation failed',
                'validation_score': validation_results['overall_score']
            }
        
        # --------------------------------------------------------
        DemoLogger.step(4, "Model Training Phase (Ultralytics YOLO)")
        
        # Initialize and train model
        self.training_service.initialize()
        training_results = self.training_service.train({
            'epochs': config['epochs'],
            'batchSize': config['batch_size']
        })
        
        # --------------------------------------------------------
        DemoLogger.step(5, "Workflow Summary")
        
        self.display_results_summary(config, validation_results, training_results)
        
        return {
            'success': True,
            'captured_frames': len(captured_frames),
            'validation_score': validation_results['overall_score'],
            'final_map': training_results['final_metrics']['mAP50'],
            'model_path': training_results['model_path']
        }
    
    def display_results_summary(self, config, validation_results, training_results):
        """Display comprehensive results summary"""
        print("\n" + "🎉 WORKFLOW COMPLETED SUCCESSFULLY!" + "\n")
        print("📊 RESULTS SUMMARY:")
        print("-" * 40)
        print(f"📝 Project: {config['project_name']}")
        print(f"📸 Frames Captured: {config['frame_count']}")
        print(f"🔍 Validation Score: {validation_results['overall_score']:.2f}/1.0")
        print(f"🧠 Training Epochs: {config['epochs']}")
        print(f"🎯 Final mAP50: {training_results['final_metrics']['mAP50']:.3f}")
        print(f"📁 Model Saved: {Path(training_results['model_path']).name}")
        
        print("\n✅ QUALITY CHECKS:")
        for check_name, result in validation_results['checks'].items():
            status_emoji = "✅" if result['status'] == 'PASS' else "⚠️"
            print(f"  {status_emoji} {check_name.replace('_', ' ').title()}: {result['score']:.2f}")
        
        print(f"\n📁 Generated Files:")
        print(f"  • Images: ./demo_data/captured_images/")
        print(f"  • Annotations: ./demo_data/annotations/")
        print(f"  • Reports: ./demo_data/validation_reports/")
        print(f"  • Models: ./demo_data/models/")
        
        if validation_results.get('recommendations'):
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in validation_results['recommendations']:
                print(f"  • {rec}")
    
    def cleanup_demo_data(self):
        """Optional cleanup of demo data"""
        response = input("\n🗑️  Clean up demo data? (y/N): ").lower()
        if response == 'y':
            import shutil
            if self.demo_data_dir.exists():
                shutil.rmtree(self.demo_data_dir)
                DemoLogger.success("Demo data cleaned up")
        else:
            DemoLogger.info(f"Demo data preserved in: {self.demo_data_dir}")

async def main():
    """Main demo entry point"""
    demo = AIModelValidationDemo()
    
    try:
        demo.display_banner()
        
        # Get user configuration
        config = demo.get_user_input()
        
        print(f"\n🚀 Starting AI Model Validation Workflow...")
        print(f"Configuration: {json.dumps(config, indent=2)}")
        
        # Run the complete workflow
        results = await demo.run_complete_workflow(config)
        
        if results['success']:
            print(f"\n🎉 SUCCESS! AI Model Validation workflow completed successfully.")
            print(f"Check the demo_data/ directory for all generated files.")
        else:
            print(f"\n❌ FAILED: {results.get('error', 'Unknown error')}")
        
        # Optional cleanup
        demo.cleanup_demo_data()
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
    except Exception as e:
        DemoLogger.error(f"Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())