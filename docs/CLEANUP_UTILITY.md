# 🧹 Automatic Cleanup Utility

## Overview

The cleanup utility automatically removes old demo output files (images and videos) to prevent disk space issues. It runs automatically when the server starts and can be configured via environment variables.

## Features

- **Automatic cleanup on server startup** - Runs every time the API server starts
- **Configurable age threshold** - Default: 30 minutes
- **Safe operation** - Only deletes files in designated demo directories
- **Dry-run mode** - Test what would be deleted without actually deleting
- **Detailed logging** - Shows exactly what files are being deleted and space freed

## Directories Cleaned

The utility monitors and cleans files from these directories:
- `demo_data/supervision_uploads/` - Uploaded video and image files
- `demo_data/captured_images/` - Webcam captured images
- `demo_data/real_integrated/` - Integrated training data
- `demo_data/annotated_videos/` - Annotated video outputs
- `camera_test_output/` - Camera test outputs
- `demo_data/supervision_camera/` - Supervision camera demos
- `runs/detect/` - YOLO detection outputs

## File Types Cleaned

Only media files with these extensions are deleted:
- **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`
- **Videos**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.m4v`, `.wmv`, `.mpg`, `.mpeg`, `.3gp`

## Configuration

### Environment Variables

- `CLEANUP_THRESHOLD_MINUTES` - Files older than this will be deleted (default: 30)
- `DISABLE_AUTO_CLEANUP` - Set to "true" to disable automatic cleanup

### Examples

```bash
# Run server with 60-minute threshold
CLEANUP_THRESHOLD_MINUTES=60 python run_api.py

# Disable automatic cleanup
DISABLE_AUTO_CLEANUP=true python run_api.py
```

## Manual Usage

### Command Line Interface

```bash
# Show help
python src/utils/cleanup_manager.py --help

# Dry run (see what would be deleted)
python src/utils/cleanup_manager.py --dry-run

# Delete files older than 60 minutes
python src/utils/cleanup_manager.py --age 60

# Dry run with custom age
python src/utils/cleanup_manager.py --age 120 --dry-run
```

### Python API

```python
from src.utils import run_cleanup

# Delete files older than 30 minutes
files_deleted, bytes_freed = run_cleanup(age_threshold_minutes=30)

# Dry run
files_deleted, bytes_freed = run_cleanup(age_threshold_minutes=60, dry_run=True)
```

## Safety Features

1. **Only designated directories** - Will not delete files outside of demo directories
2. **Only media files** - Only deletes image and video files
3. **Age-based** - Only deletes files older than threshold
4. **Non-critical failures** - Cleanup failures don't stop the server
5. **Detailed logging** - All operations are logged

## Example Output

```
============================================================
🧹 Demo Output Cleanup Utility
============================================================
Threshold: 30 minutes
Mode: LIVE
============================================================
2025-08-01 15:34:04,479 - INFO - Starting cleanup (threshold: 30 minutes)
2025-08-01 15:34:04,481 - INFO - Found 46 files to delete
2025-08-01 15:34:04,481 - INFO - Deleting: camera_test_output/test_frame_1.jpg (age: 13.7 hours, size: 5.3 KB)
...
2025-08-01 15:34:04,493 - INFO - Cleanup Summary:
2025-08-01 15:34:04,493 - INFO - Files deleted: 46
2025-08-01 15:34:04,493 - INFO - Space freed: 477.6 MB
```

## Troubleshooting

### Cleanup not running

1. Check if `DISABLE_AUTO_CLEANUP` is set to "true"
2. Check server logs for cleanup errors
3. Verify directories exist and have write permissions

### Files not being deleted

1. Check file age - must be older than threshold
2. Verify file extension is in the cleanup list
3. Check file permissions

### Performance impact

The cleanup runs asynchronously on startup and typically completes in seconds. For very large directories with thousands of files, you may want to:
- Run cleanup manually during off-hours
- Increase the age threshold
- Disable auto-cleanup and use cron/scheduled tasks

## Best Practices

1. **Monitor disk space** - Check available space regularly
2. **Adjust threshold** - Set based on your demo usage patterns
3. **Archive important files** - Move files you want to keep to permanent storage
4. **Use .gitignore** - Ensure demo directories are excluded from git

## Integration with CI/CD

For automated deployments, you can:

```yaml
# docker-compose.yml
services:
  api:
    environment:
      - CLEANUP_THRESHOLD_MINUTES=60
      - DISABLE_AUTO_CLEANUP=false
```

```dockerfile
# Dockerfile
ENV CLEANUP_THRESHOLD_MINUTES=30
ENV DISABLE_AUTO_CLEANUP=false
```