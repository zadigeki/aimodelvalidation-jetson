// Core types for the supervision interface

export interface ValidationFile {
  id: string;
  name: string;
  type: 'image' | 'video';
  size: number;
  url: string;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  uploadProgress?: number;
  processingProgress?: number;
  metadata?: FileMetadata;
  results?: ValidationResults;
  createdAt: Date;
  updatedAt: Date;
}

export interface FileMetadata {
  width?: number;
  height?: number;
  duration?: number; // for videos
  frameRate?: number;
  codec?: string;
  bitrate?: number;
  fileFormat: string;
  thumbnailUrl?: string;
}

export interface ValidationResults {
  confidence: number;
  objects: DetectedObject[];
  annotations: Annotation[];
  frames?: FrameResult[]; // for videos
  summary: ValidationSummary;
  processedAt: Date;
}

export interface DetectedObject {
  id: string;
  class: string;
  confidence: number;
  bbox: BoundingBox;
  frame?: number; // for video objects
  timestamp?: number; // for video objects
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Annotation {
  id: string;
  type: 'detection' | 'classification' | 'segmentation';
  class: string;
  confidence: number;
  bbox?: BoundingBox;
  points?: Point[]; // for segmentation
  frame?: number;
  timestamp?: number;
  verified: boolean;
  createdBy: string;
  createdAt: Date;
}

export interface Point {
  x: number;
  y: number;
}

export interface FrameResult {
  frameNumber: number;
  timestamp: number;
  objects: DetectedObject[];
  thumbnailUrl: string;
  confidence: number;
}

export interface ValidationSummary {
  totalObjects: number;
  classDistribution: { [className: string]: number };
  averageConfidence: number;
  processingTime: number;
  frameCount?: number;
  qualityScore: number;
}

export interface ValidationProgress {
  fileId: string;
  stage: 'upload' | 'preprocessing' | 'detection' | 'annotation' | 'validation' | 'complete';
  progress: number; // 0-100
  message: string;
  timestamp: Date;
  estimatedTimeRemaining?: number;
}

export interface ValidationReport {
  id: string;
  name: string;
  files: ValidationFile[];
  summary: ReportSummary;
  createdAt: Date;
  generatedBy: string;
  format: 'json' | 'csv' | 'pdf' | 'xml';
}

export interface ReportSummary {
  totalFiles: number;
  totalObjects: number;
  averageConfidence: number;
  qualityDistribution: {
    high: number;
    medium: number;
    low: number;
  };
  classBreakdown: { [className: string]: ClassStats };
}

export interface ClassStats {
  count: number;
  averageConfidence: number;
  distribution: number[];
}

export interface VideoPlayerState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  playbackRate: number;
  volume: number;
  muted: boolean;
}

export interface AppState {
  files: ValidationFile[];
  selectedFile: ValidationFile | null;
  validationProgress: { [fileId: string]: ValidationProgress };
  reports: ValidationReport[];
  settings: AppSettings;
  user: User | null;
}

export interface AppSettings {
  autoPlay: boolean;
  showConfidenceScores: boolean;
  confidenceThreshold: number;
  darkMode: boolean;
  notifications: boolean;
  autoExport: boolean;
  exportFormat: 'json' | 'csv' | 'pdf';
}

export interface User {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'annotator' | 'viewer';
  permissions: string[];
}

export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface UploadOptions {
  maxFileSize?: number;
  acceptedTypes?: string[];
  compressionQuality?: number;
  generateThumbnails?: boolean;
}

export interface WebSocketMessage {
  type: 'progress' | 'result' | 'error' | 'connection';
  fileId?: string;
  data: any;
  timestamp: Date;
}

export interface ConfidenceConfig {
  threshold: number;
  colors: {
    high: string;
    medium: string;
    low: string;
  };
}

export interface ExportOptions {
  format: 'json' | 'csv' | 'pdf' | 'xml';
  includeImages: boolean;
  includeAnnotations: boolean;
  includeMetadata: boolean;
  confidenceThreshold?: number;
  dateRange?: {
    start: Date;
    end: Date;
  };
}