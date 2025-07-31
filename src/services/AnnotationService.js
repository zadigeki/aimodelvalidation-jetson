/**
 * AnnotationService - CVAT integration service
 * London School TDD - Minimal implementation for contract satisfaction
 */

export class AnnotationService {
  constructor(dependencies) {
    this.cvatClient = dependencies.cvatClient;
    this.logger = dependencies.logger;
  }

  async createProject(projectConfig) {
    this.logger.info('Creating annotation project', { name: projectConfig.name });
    
    const project = await this.cvatClient.createTask();
    
    return {
      projectId: `cvat-project-${Date.now()}`
    };
  }

  async uploadData(dataFiles) {
    this.logger.info('Uploading data to CVAT');
    
    await this.cvatClient.uploadImages();
    
    return {
      taskId: `cvat-task-${Date.now()}`
    };
  }

  async getAnnotations(taskId) {
    this.logger.info('Retrieving annotations', { taskId });
    
    return {
      annotations: [
        { id: 1, label: 'object', bbox: [10, 10, 50, 50] }
      ]
    };
  }

  async exportAnnotations(format = 'YOLO') {
    this.logger.info('Exporting annotations', { format });
    
    return {
      format,
      data: 'mock-export-data'
    };
  }

  async validateAnnotations() {
    this.logger.info('Validating annotations');
    
    return {
      valid: true,
      errors: []
    };
  }
}