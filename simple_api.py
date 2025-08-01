"""Simple FastAPI backend for Supervision UI with Real Object Detection"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import shutil
from pathlib import Path
import uuid
from datetime import datetime
import asyncio
from typing import List, Dict, Any
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import time
import sys
import os

# Add src to path for cleanup utility
sys.path.insert(0, str(Path(__file__).parent / "src"))
try:
    from utils import auto_cleanup_on_startup
except ImportError:
    auto_cleanup_on_startup = None

app = FastAPI(title="AI Model Validation API")

# Configure CORS - Allow all origins for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins including file:// URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage paths
UPLOAD_DIR = Path("demo_data/supervision_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory storage for demo
uploaded_files: Dict[str, Dict[str, Any]] = {}

# Initialize YOLO model and Supervision components
model = None
box_annotator = None

def initialize_model():
    """Initialize YOLO model and Supervision components"""
    global model, box_annotator
    try:
        print("🤖 Loading YOLO model...")
        model = YOLO('yolov8n.pt')  # Using nano model for speed
        box_annotator = sv.BoxAnnotator(thickness=2)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None
        box_annotator = None

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("\n" + "="*60)
    print("🚀 Starting AI Model Validation API")
    print("="*60)
    
    # Run cleanup if available
    if auto_cleanup_on_startup:
        try:
            print("🧹 Running automatic cleanup of old demo files...")
            auto_cleanup_on_startup()
        except Exception as e:
            print(f"⚠️  Cleanup failed (non-critical): {e}")
    
    # Initialize model
    initialize_model()
    
    print("="*60)
    print("✅ Server ready!")
    print("="*60 + "\n")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the upload interface"""
    html_file = Path("simple_upload_demo.html")
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(), status_code=200)
    return {"message": "AI Model Validation API with Supervision Integration"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a video or image file"""
    try:
        # Generate unique ID
        file_id = str(uuid.uuid4())
        
        # Save file
        file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Store metadata
        uploaded_files[file_id] = {
            "id": file_id,
            "filename": file.filename,
            "path": str(file_path),
            "size": file_path.stat().st_size,
            "type": "video" if file.filename.endswith(('.mp4', '.avi', '.mov')) else "image",
            "uploaded_at": datetime.now().isoformat(),
            "status": "uploaded",
            "progress": 0
        }
        
        return JSONResponse(content={
            "id": file_id,
            "filename": file.filename,
            "message": "File uploaded successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files")
async def list_files():
    """List all uploaded files"""
    return list(uploaded_files.values())

@app.get("/api/files/{file_id}")
async def get_file(file_id: str):
    """Get file details"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    return uploaded_files[file_id]

@app.post("/api/files/{file_id}/validate")
async def validate_file(file_id: str):
    """Start validation process for a file"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = uploaded_files[file_id]
    file_info["status"] = "processing"
    
    # Start real processing with Supervision
    asyncio.create_task(process_with_supervision(file_id))
    
    return {"message": "Validation started", "file_id": file_id}

async def process_with_supervision(file_id: str):
    """Process file with real Supervision object detection"""
    global model, box_annotator
    
    file_info = uploaded_files[file_id]
    file_path = Path(file_info["path"])
    
    try:
        start_time = time.time()
        
        # Update progress
        file_info["progress"] = 20
        
        if model is None:
            initialize_model()
            if model is None:
                raise Exception("Model failed to load")
        
        file_info["progress"] = 40
        
        if file_info["type"] == "image":
            results = await process_image(file_path)
        else:
            results = await process_video(file_path, file_info)
            
        processing_time = time.time() - start_time
        
        file_info["status"] = "completed"
        file_info["progress"] = 100
        file_info["results"] = {
            **results,
            "processing_time": f"{processing_time:.1f}s"
        }
        
    except Exception as e:
        print(f"❌ Processing failed for {file_id}: {e}")
        file_info["status"] = "failed"
        file_info["error"] = str(e)

async def process_image(file_path: Path):
    """Process single image with object detection"""
    # Load image
    image = cv2.imread(str(file_path))
    if image is None:
        raise Exception("Could not load image")
    
    # Run detection
    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Extract results
    classes = [model.names[class_id] for class_id in detections.class_id] if len(detections) > 0 else []
    unique_classes = list(set(classes))
    avg_confidence = float(np.mean(detections.confidence)) if len(detections) > 0 else 0.0
    
    return {
        "objects_detected": len(detections),
        "classes": unique_classes,
        "confidence_avg": avg_confidence,
        "quality_score": min(avg_confidence + 0.1, 1.0)  # Simple quality metric
    }

async def process_video(file_path: Path, file_info: Dict):
    """Process video with frame-by-frame object detection"""
    cap = cv2.VideoCapture(str(file_path))
    
    if not cap.isOpened():
        raise Exception("Could not open video file")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    all_detections = []
    detailed_report = []
    frame_count = 0
    
    # Process every 10th frame for efficiency
    frame_skip = max(1, total_frames // 20)  # Process ~20 frames max
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            # Run detection on this frame
            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            all_detections.append(detections)
            
            # Create detailed frame report
            timestamp = frame_count / fps if fps > 0 else 0
            frame_objects = []
            
            if len(detections) > 0:
                for i in range(len(detections)):
                    class_name = model.names[detections.class_id[i]]
                    confidence = float(detections.confidence[i])
                    bbox = detections.xyxy[i]  # [x1, y1, x2, y2]
                    
                    frame_objects.append({
                        "class": class_name,
                        "confidence": round(confidence, 3),
                        "bbox": [float(x) for x in bbox]
                    })
            
            detailed_report.append({
                "frame": frame_count,
                "timestamp": round(timestamp, 2),
                "objects_count": len(detections),
                "objects": frame_objects
            })
            
            # Update progress
            progress = 60 + (frame_count / total_frames) * 30
            file_info["progress"] = int(progress)
        
        frame_count += 1
    
    cap.release()
    
    # Combine all detections
    all_classes = []
    all_confidences = []
    
    for detections in all_detections:
        if len(detections) > 0:
            classes = [model.names[class_id] for class_id in detections.class_id]
            all_classes.extend(classes)
            all_confidences.extend(detections.confidence)
    
    if all_classes:
        unique_classes = list(set(all_classes))
        avg_confidence = float(np.mean(all_confidences))
        total_objects = len(all_classes)
    else:
        unique_classes = []
        avg_confidence = 0.0
        total_objects = 0
    
    # Store detailed report in file info for later retrieval
    file_info["detailed_report"] = detailed_report
    
    return {
        "objects_detected": total_objects,
        "classes": unique_classes,
        "confidence_avg": avg_confidence,
        "quality_score": min(avg_confidence + 0.05, 1.0),
        "total_frames": total_frames,
        "fps": fps,
        "frames_processed": len(detailed_report)
    }

@app.get("/api/files/{file_id}/results")
async def get_results(file_id: str):
    """Get validation results"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = uploaded_files[file_id]
    if file_info["status"] != "completed":
        return {"status": file_info["status"], "progress": file_info.get("progress", 0)}
    
    return file_info.get("results", {})

@app.get("/api/files/{file_id}/progress")
async def get_progress(file_id: str):
    """Get validation progress"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = uploaded_files[file_id]
    return {
        "status": file_info["status"],
        "progress": file_info.get("progress", 0)
    }

@app.get("/api/files/{file_id}/report")
async def get_detailed_report(file_id: str):
    """Get detailed frame-by-frame detection report"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = uploaded_files[file_id]
    
    if file_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="File processing not completed")
    
    detailed_report = file_info.get("detailed_report", [])
    
    return {
        "file_id": file_id,
        "filename": file_info["filename"],
        "total_frames": file_info.get("results", {}).get("total_frames", 0),
        "fps": file_info.get("results", {}).get("fps", 0),
        "processing_time": file_info.get("results", {}).get("processing_time", "0s"),
        "summary": file_info.get("results", {}),
        "frame_detections": detailed_report
    }

@app.get("/api/files/{file_id}/report/markdown")
async def get_markdown_report(file_id: str):
    """Generate detailed markdown evaluation report"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = uploaded_files[file_id]
    
    if file_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="File processing not completed")
    
    # Generate markdown report
    markdown_content = generate_evaluation_report(file_info)
    
    return {"markdown": markdown_content}

@app.get("/api/files/{file_id}/report/download")
async def download_markdown_report(file_id: str):
    """Download markdown evaluation report as .md file"""
    from fastapi.responses import Response
    
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = uploaded_files[file_id]
    
    if file_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="File processing not completed")
    
    # Generate markdown content
    markdown_content = generate_evaluation_report(file_info)
    
    # Create filename
    filename = f"evaluation_report_{file_info['filename']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    # Save to reports directory
    reports_dir = Path("demo_data/evaluation_reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = reports_dir / filename
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    # Return file for download
    return Response(
        content=markdown_content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Type": "text/markdown; charset=utf-8"
        }
    )

@app.post("/api/files/{file_id}/annotate")
async def create_annotated_video(file_id: str):
    """Generate annotated video with bounding boxes"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = uploaded_files[file_id]
    
    if file_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="File processing not completed")
    
    if file_info["type"] != "video":
        raise HTTPException(status_code=400, detail="Only video files can be annotated")
    
    # Start annotation process
    file_info["annotation_status"] = "processing"
    file_info["annotation_progress"] = 0
    
    asyncio.create_task(generate_annotated_video(file_id))
    
    return {"message": "Video annotation started", "file_id": file_id}

@app.get("/api/files/{file_id}/annotate/progress")
async def get_annotation_progress(file_id: str):
    """Get video annotation progress"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = uploaded_files[file_id]
    return {
        "status": file_info.get("annotation_status", "not_started"),
        "progress": file_info.get("annotation_progress", 0),
        "output_file": file_info.get("annotated_video_path", None)
    }

@app.get("/api/files/{file_id}/annotate/download")
async def download_annotated_video(file_id: str):
    """Download annotated video file"""
    from fastapi.responses import FileResponse
    
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = uploaded_files[file_id]
    
    if file_info.get("annotation_status") != "completed":
        raise HTTPException(status_code=400, detail="Video annotation not completed")
    
    annotated_path = file_info.get("annotated_video_path")
    if not annotated_path or not Path(annotated_path).exists():
        raise HTTPException(status_code=404, detail="Annotated video file not found")
    
    return FileResponse(
        path=annotated_path,
        filename=f"annotated_{file_info['filename']}",
        media_type="video/mp4"
    )

async def generate_annotated_video(file_id: str):
    """Generate video with bounding box annotations"""
    global model, box_annotator
    
    file_info = uploaded_files[file_id]
    input_path = Path(file_info["path"])
    detailed_report = file_info.get("detailed_report", [])
    
    try:
        # Create output directory
        output_dir = Path("demo_data/annotated_videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output filename
        output_filename = f"annotated_{input_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = output_dir / output_filename
        
        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise Exception("Could not open input video")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Create detection lookup for faster access
        detection_lookup = {}
        for frame_data in detailed_report:
            frame_num = frame_data["frame"]
            detection_lookup[frame_num] = frame_data.get("objects", [])
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if this frame has detections
            if frame_count in detection_lookup:
                objects = detection_lookup[frame_count]
                
                # Create annotations for this frame
                for obj in objects:
                    bbox = obj["bbox"]
                    class_name = obj["class"]
                    confidence = obj["confidence"]
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Choose color based on confidence
                    if confidence >= 0.8:
                        color = (0, 255, 0)  # Green - High confidence
                    elif confidence >= 0.5:
                        color = (0, 255, 255)  # Yellow - Medium confidence
                    else:
                        color = (0, 0, 255)  # Red - Low confidence
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label with class and confidence
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add frame info overlay
                timestamp = frame_count / fps if fps > 0 else 0
                minutes = int(timestamp // 60)
                seconds = timestamp % 60
                frame_info = f"Frame {frame_count} | {minutes:02d}:{seconds:05.2f} | Objects: {len(objects)}"
                
                cv2.putText(frame, frame_info, (10, height - 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, frame_info, (10, height - 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Write frame to output video
            out.write(frame)
            
            # Update progress
            progress = int((frame_count / total_frames) * 100)
            file_info["annotation_progress"] = progress
            
            frame_count += 1
        
        # Clean up
        cap.release()
        out.release()
        
        # Update file info
        file_info["annotation_status"] = "completed"
        file_info["annotation_progress"] = 100
        file_info["annotated_video_path"] = str(output_path)
        
        print(f"✅ Annotated video created: {output_path}")
        
    except Exception as e:
        print(f"❌ Video annotation failed: {e}")
        file_info["annotation_status"] = "failed"
        file_info["annotation_error"] = str(e)

def generate_evaluation_report(file_info: Dict) -> str:
    """Generate comprehensive markdown evaluation report"""
    from datetime import datetime
    
    results = file_info.get("results", {})
    detailed_report = file_info.get("detailed_report", [])
    
    # Calculate statistics
    total_detections = sum(frame.get("objects_count", 0) for frame in detailed_report)
    confidence_scores = []
    class_counts = {}
    
    for frame in detailed_report:
        for obj in frame.get("objects", []):
            confidence_scores.append(obj["confidence"])
            class_name = obj["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    # Generate markdown
    markdown = f"""# AI Object Detection Evaluation Report

## 📋 Executive Summary

**File:** `{file_info["filename"]}`  
**Evaluation Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**File ID:** `{file_info["id"]}`  
**Processing Status:** ✅ Completed  

---

## 🎬 Video Specifications

| Attribute | Value |
|-----------|-------|
| **File Size** | {file_info.get("size", 0) / (1024*1024):.1f} MB |
| **Total Frames** | {results.get("total_frames", "N/A")} |
| **Frame Rate (FPS)** | {results.get("fps", "N/A"):.2f} |
| **Duration** | {results.get("total_frames", 0) / results.get("fps", 1):.2f} seconds |
| **Processing Time** | {results.get("processing_time", "N/A")} |
| **Frames Analyzed** | {len(detailed_report)} |
| **Analysis Coverage** | {(len(detailed_report) / results.get("total_frames", 1) * 100):.1f}% |

---

## 🎯 Detection Summary

### Overall Results
- **Total Objects Detected:** {total_detections}
- **Unique Object Classes:** {len(results.get("classes", []))}
- **Average Confidence Score:** {avg_confidence:.3f} ({avg_confidence*100:.1f}%)
- **Quality Assessment:** {results.get("quality_score", 0)*100:.1f}%

### Object Class Distribution
"""
    
    # Add class distribution table
    if class_counts:
        markdown += "\n| Object Class | Count | Percentage |\n|--------------|-------|------------|\n"
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            markdown += f"| **{class_name}** | {count} | {percentage:.1f}% |\n"
    
    markdown += f"""

### Confidence Score Analysis
- **Minimum Confidence:** {min(confidence_scores):.3f} ({min(confidence_scores)*100:.1f}%)
- **Maximum Confidence:** {max(confidence_scores):.3f} ({max(confidence_scores)*100:.1f}%)
- **Standard Deviation:** {(sum((x - avg_confidence)**2 for x in confidence_scores) / len(confidence_scores))**0.5:.3f}

---

## 🔍 Frame-by-Frame Analysis

**Note for Independent Evaluators:** Each frame entry below shows the exact timestamp, detected objects with confidence scores, and bounding box coordinates. Confidence scores range from 0.0 to 1.0, where higher values indicate greater certainty.

"""

    # Add detailed frame analysis
    for i, frame in enumerate(detailed_report):
        timestamp = frame["timestamp"]
        frame_num = frame["frame"]
        objects = frame.get("objects", [])
        
        minutes = int(timestamp // 60)
        seconds = timestamp % 60
        
        markdown += f"""### Frame {frame_num} - {minutes:02d}:{seconds:05.2f}

**Timestamp:** {timestamp:.2f}s  
**Objects Detected:** {len(objects)}  

"""
        
        if objects:
            markdown += """| Object | Confidence | Bounding Box (x1,y1,x2,y2) | Assessment |
|--------|------------|----------------------------|------------|
"""
            for obj in objects:
                conf = obj["confidence"]
                bbox = obj["bbox"]
                
                # Assessment based on confidence
                if conf >= 0.8:
                    assessment = "🟢 High Confidence"
                elif conf >= 0.5:
                    assessment = "🟡 Medium Confidence"
                else:
                    assessment = "🔴 Low Confidence"
                
                bbox_str = f"({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})"
                markdown += f"| **{obj['class']}** | {conf:.3f} ({conf*100:.1f}%) | {bbox_str} | {assessment} |\n"
        else:
            markdown += "*No objects detected in this frame.*\n"
        
        markdown += "\n"
    
    markdown += f"""---

## 📊 Technical Validation

### Model Information
- **AI Model:** YOLOv8 Nano (yolov8n.pt)
- **Model Size:** 3,006,038 parameters
- **Framework:** Ultralytics + Roboflow Supervision
- **Processing Backend:** FastAPI + OpenCV

### Processing Parameters
- **Confidence Threshold:** Default YOLOv8 threshold
- **Frame Sampling:** Every {results.get("total_frames", 1) // len(detailed_report) if len(detailed_report) > 0 else 1} frames
- **Input Resolution:** 640x384 (model default)
- **Preprocessing:** Automatic scaling and normalization

### Quality Metrics
- **Detection Consistency:** {len([f for f in detailed_report if f.get("objects_count", 0) > 0])} / {len(detailed_report)} frames with detections
- **Average Objects per Frame:** {total_detections / len(detailed_report):.2f}
- **Confidence Distribution:** {len([c for c in confidence_scores if c >= 0.8])} high ({len([c for c in confidence_scores if c >= 0.8])/len(confidence_scores)*100:.1f}%), {len([c for c in confidence_scores if 0.5 <= c < 0.8])} medium ({len([c for c in confidence_scores if 0.5 <= c < 0.8])/len(confidence_scores)*100:.1f}%), {len([c for c in confidence_scores if c < 0.5])} low ({len([c for c in confidence_scores if c < 0.5])/len(confidence_scores)*100:.1f}%)

---

## 🎯 Evaluation Guidelines for Independent Assessors

### How to Validate These Results

1. **Timestamp Verification**
   - Use video player to navigate to specified timestamps
   - Compare detected objects with actual video content
   - Verify object classifications are accurate

2. **Bounding Box Assessment**
   - Coordinates are in pixels from top-left corner
   - Format: (x1, y1, x2, y2) where (x1,y1) is top-left, (x2,y2) is bottom-right
   - Check if bounding boxes properly encompass detected objects

3. **Confidence Score Interpretation**
   - Scores ≥ 0.8: Very likely correct detection
   - Scores 0.5-0.8: Likely correct, manual verification recommended
   - Scores < 0.5: Low confidence, high chance of false positive

4. **Classification Accuracy**
   - Verify object class labels match visual appearance
   - Note any obvious misclassifications
   - Consider lighting, angle, and occlusion effects

### Expected Detection Capabilities
- **Strong Performance:** Cars, trucks, buses (common training data)
- **Good Performance:** Boats, trains (less common but well-represented)
- **Variable Performance:** Partially occluded or distant objects
- **Limitations:** Very small objects, unusual angles, poor lighting

---

## 📈 Recommendations

### For Production Use
1. **Confidence Filtering:** Consider filtering detections below 0.5 confidence
2. **Temporal Smoothing:** Implement tracking to reduce frame-to-frame inconsistencies
3. **Class-Specific Tuning:** Adjust thresholds per object class based on use case
4. **Validation Dataset:** Create ground truth annotations for systematic evaluation

### For Accuracy Improvement
1. **Model Upgrade:** Consider YOLOv8s or YOLOv8m for higher accuracy
2. **Custom Training:** Fine-tune on domain-specific data if available
3. **Ensemble Methods:** Combine multiple models for improved reliability
4. **Post-Processing:** Implement non-maximum suppression and tracking

---

**Report Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} UTC  
**System:** AI Model Validation Platform with Roboflow Supervision  
**Version:** 1.0.0  

---

*This report provides objective, technical analysis of AI model performance. For questions about methodology or validation procedures, consult the technical documentation.*
"""
    
    return markdown

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Model Validation API")
    print("📡 Server: http://localhost:8000")
    print("📚 Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/api/health")
    
    # Initialize model at startup
    initialize_model()
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")