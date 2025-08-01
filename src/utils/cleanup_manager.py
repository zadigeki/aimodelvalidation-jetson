#!/usr/bin/env python3
"""
Cleanup Manager - Automatically removes old demo output files to save disk space

This utility deletes image and video files older than a specified time threshold
from demo output directories to prevent disk space issues.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CleanupManager:
    """Manages cleanup of old demo output files"""
    
    # Directories to clean (relative to project root)
    CLEANUP_DIRECTORIES = [
        "demo_data/supervision_uploads",
        "demo_data/captured_images",
        "demo_data/real_integrated",
        "demo_data/annotated_videos",  # Add annotated videos directory
        "camera_test_output",
        "demo_data/supervision_camera",
        "runs/detect",
    ]
    
    # File extensions to clean
    CLEANUP_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp',  # Images
        '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv',   # Videos
        '.m4v', '.wmv', '.mpg', '.mpeg', '.3gp'            # More videos
    }
    
    def __init__(self, age_threshold_minutes: int = 30, dry_run: bool = False):
        """
        Initialize cleanup manager
        
        Args:
            age_threshold_minutes: Files older than this will be deleted (default: 30)
            dry_run: If True, only log what would be deleted without actually deleting
        """
        self.age_threshold_minutes = age_threshold_minutes
        self.dry_run = dry_run
        self.project_root = Path(__file__).parent.parent.parent
        
    def get_file_age_minutes(self, file_path: Path) -> float:
        """Get file age in minutes"""
        try:
            file_stat = file_path.stat()
            file_modified_time = file_stat.st_mtime
            current_time = time.time()
            age_seconds = current_time - file_modified_time
            return age_seconds / 60
        except Exception as e:
            logger.error(f"Error getting file age for {file_path}: {e}")
            return 0
    
    def format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def find_old_files(self) -> List[Tuple[Path, float, int]]:
        """
        Find all files older than threshold
        
        Returns:
            List of tuples (file_path, age_minutes, size_bytes)
        """
        old_files = []
        
        for directory in self.CLEANUP_DIRECTORIES:
            dir_path = self.project_root / directory
            
            if not dir_path.exists():
                logger.debug(f"Directory does not exist: {dir_path}")
                continue
                
            logger.info(f"Scanning directory: {dir_path}")
            
            try:
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in self.CLEANUP_EXTENSIONS:
                        age_minutes = self.get_file_age_minutes(file_path)
                        
                        if age_minutes > self.age_threshold_minutes:
                            size_bytes = file_path.stat().st_size
                            old_files.append((file_path, age_minutes, size_bytes))
                            
            except Exception as e:
                logger.error(f"Error scanning directory {dir_path}: {e}")
                
        return old_files
    
    def cleanup(self) -> Tuple[int, int]:
        """
        Perform cleanup of old files
        
        Returns:
            Tuple of (files_deleted, total_size_freed)
        """
        logger.info(f"Starting cleanup (threshold: {self.age_threshold_minutes} minutes)")
        if self.dry_run:
            logger.info("DRY RUN MODE - No files will be deleted")
        
        old_files = self.find_old_files()
        
        if not old_files:
            logger.info("No old files found to delete")
            return 0, 0
        
        # Sort by age (oldest first)
        old_files.sort(key=lambda x: x[1], reverse=True)
        
        files_deleted = 0
        total_size_freed = 0
        
        logger.info(f"Found {len(old_files)} files to delete")
        
        for file_path, age_minutes, size_bytes in old_files:
            age_hours = age_minutes / 60
            relative_path = file_path.relative_to(self.project_root)
            
            log_msg = f"{'Would delete' if self.dry_run else 'Deleting'}: {relative_path} " \
                     f"(age: {age_hours:.1f} hours, size: {self.format_size(size_bytes)})"
            logger.info(log_msg)
            
            if not self.dry_run:
                try:
                    file_path.unlink()
                    files_deleted += 1
                    total_size_freed += size_bytes
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
        
        # Summary
        logger.info(f"\nCleanup Summary:")
        logger.info(f"Files {'would be' if self.dry_run else ''} deleted: {files_deleted}")
        logger.info(f"Space {'would be' if self.dry_run else ''} freed: {self.format_size(total_size_freed)}")
        
        return files_deleted, total_size_freed


def run_cleanup(age_threshold_minutes: int = 30, dry_run: bool = False):
    """
    Run cleanup with specified parameters
    
    Args:
        age_threshold_minutes: Delete files older than this
        dry_run: If True, only show what would be deleted
    """
    manager = CleanupManager(age_threshold_minutes, dry_run)
    return manager.cleanup()


def auto_cleanup_on_startup():
    """
    Run automatic cleanup on server startup
    This function is called from the main API startup
    """
    logger.info("Running automatic cleanup on server startup...")
    
    # Check if cleanup is disabled via environment variable
    if os.getenv("DISABLE_AUTO_CLEANUP", "false").lower() == "true":
        logger.info("Auto cleanup is disabled via DISABLE_AUTO_CLEANUP environment variable")
        return
    
    # Get threshold from environment or use default
    threshold = int(os.getenv("CLEANUP_THRESHOLD_MINUTES", "30"))
    
    # Run cleanup
    files_deleted, size_freed = run_cleanup(age_threshold_minutes=threshold)
    
    logger.info(f"Startup cleanup completed: {files_deleted} files deleted, "
                f"{size_freed / (1024*1024):.1f} MB freed")


if __name__ == "__main__":
    # Command line usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up old demo output files")
    parser.add_argument(
        "--age", 
        type=int, 
        default=30,
        help="Delete files older than this many minutes (default: 30)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"🧹 Demo Output Cleanup Utility")
    print(f"{'='*60}")
    print(f"Threshold: {args.age} minutes")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"{'='*60}\n")
    
    run_cleanup(age_threshold_minutes=args.age, dry_run=args.dry_run)