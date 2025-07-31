"""Dependency injection container for testable architecture"""

from typing import Dict, Any, TypeVar, Type, Callable, Optional, get_type_hints
from dataclasses import dataclass
import inspect
import asyncio
from pathlib import Path

T = TypeVar('T')

class LifetimeScope:
    """Scopes for service lifetimes"""
    TRANSIENT = "transient"  # New instance each time
    SINGLETON = "singleton"  # Same instance each time
    SCOPED = "scoped"        # Same instance within scope

@dataclass
class ServiceRegistration:
    """Registration information for a service"""
    service_type: Type
    implementation: Type | Callable
    lifetime: str = LifetimeScope.TRANSIENT
    name: Optional[str] = None
    
class ServiceScope:
    """Service scope for managing scoped instances"""
    
    def __init__(self):
        self._scoped_instances: Dict[Type, Any] = {}
    
    def get_scoped_instance(self, service_type: Type) -> Optional[Any]:
        """Get scoped instance if exists"""
        return self._scoped_instances.get(service_type)
    
    def set_scoped_instance(self, service_type: Type, instance: Any) -> None:
        """Set scoped instance"""
        self._scoped_instances[service_type] = instance
    
    def dispose(self) -> None:
        """Dispose all scoped instances"""
        for instance in self._scoped_instances.values():
            if hasattr(instance, 'dispose'):
                instance.dispose()
        self._scoped_instances.clear()

class Container:
    """Dependency injection container"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceRegistration] = {}
        self._singletons: Dict[Type, Any] = {}
        self._current_scope: Optional[ServiceScope] = None
        
    def register_transient(self, service_type: Type[T], implementation: Type[T] | Callable[[], T], name: Optional[str] = None) -> None:
        """Register transient service (new instance each time)
        
        Args:
            service_type: Service interface type
            implementation: Implementation type or factory function
            name: Optional service name for named registrations
        """
        self._services[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation=implementation,
            lifetime=LifetimeScope.TRANSIENT,
            name=name
        )
    
    def register_singleton(self, service_type: Type[T], implementation: Type[T] | Callable[[], T], name: Optional[str] = None) -> None:
        """Register singleton service (same instance each time)
        
        Args:
            service_type: Service interface type
            implementation: Implementation type or factory function
            name: Optional service name for named registrations
        """
        self._services[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation=implementation,
            lifetime=LifetimeScope.SINGLETON,
            name=name
        )
    
    def register_scoped(self, service_type: Type[T], implementation: Type[T] | Callable[[], T], name: Optional[str] = None) -> None:
        """Register scoped service (same instance within scope)
        
        Args:
            service_type: Service interface type
            implementation: Implementation type or factory function
            name: Optional service name for named registrations
        """
        self._services[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation=implementation,
            lifetime=LifetimeScope.SCOPED,
            name=name
        )
    
    def register_instance(self, service_type: Type[T], instance: T, name: Optional[str] = None) -> None:
        """Register existing instance as singleton
        
        Args:
            service_type: Service interface type
            instance: Service instance
            name: Optional service name for named registrations
        """
        self._services[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation=lambda: instance,
            lifetime=LifetimeScope.SINGLETON,
            name=name
        )
        self._singletons[service_type] = instance
    
    def register_factory(self, service_type: Type[T], factory: Callable[['Container'], T], lifetime: str = LifetimeScope.TRANSIENT, name: Optional[str] = None) -> None:
        """Register factory function for service
        
        Args:
            service_type: Service interface type
            factory: Factory function that takes container as parameter
            lifetime: Service lifetime
            name: Optional service name for named registrations
        """
        self._services[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation=lambda: factory(self),
            lifetime=lifetime,
            name=name
        )
    
    def resolve(self, service_type: Type[T], name: Optional[str] = None) -> T:
        """Resolve service instance
        
        Args:
            service_type: Service type to resolve
            name: Optional service name for named resolution
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service not registered
            TypeError: If service cannot be instantiated
        """
        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} not registered")
        
        registration = self._services[service_type]
        
        # Check for named registration mismatch
        if name != registration.name:
            raise ValueError(f"Service {service_type.__name__} registered with different name")
        
        # Return singleton if already created
        if registration.lifetime == LifetimeScope.SINGLETON and service_type in self._singletons:
            return self._singletons[service_type]
        
        # Return scoped instance if exists
        if registration.lifetime == LifetimeScope.SCOPED and self._current_scope:
            scoped_instance = self._current_scope.get_scoped_instance(service_type)
            if scoped_instance is not None:
                return scoped_instance
        
        # Create new instance
        try:
            instance = self._create_instance(registration)
        except Exception as e:
            raise TypeError(f"Failed to create instance of {service_type.__name__}: {e}") from e
        
        # Store based on lifetime
        if registration.lifetime == LifetimeScope.SINGLETON:
            self._singletons[service_type] = instance
        elif registration.lifetime == LifetimeScope.SCOPED and self._current_scope:
            self._current_scope.set_scoped_instance(service_type, instance)
        
        return instance
    
    def try_resolve(self, service_type: Type[T], name: Optional[str] = None) -> Optional[T]:
        """Try to resolve service, return None if not registered
        
        Args:
            service_type: Service type to resolve
            name: Optional service name
            
        Returns:
            Service instance or None if not registered
        """
        try:
            return self.resolve(service_type, name)
        except (ValueError, TypeError):
            return None
    
    def is_registered(self, service_type: Type, name: Optional[str] = None) -> bool:
        """Check if service is registered
        
        Args:
            service_type: Service type to check
            name: Optional service name
            
        Returns:
            True if service is registered
        """
        if service_type not in self._services:
            return False
        
        registration = self._services[service_type]
        return name == registration.name
    
    def create_scope(self) -> ServiceScope:
        """Create new service scope
        
        Returns:
            New service scope
        """
        return ServiceScope()
    
    def enter_scope(self, scope: ServiceScope) -> None:
        """Enter service scope
        
        Args:
            scope: Scope to enter
        """
        self._current_scope = scope
    
    def exit_scope(self) -> None:
        """Exit current service scope"""
        if self._current_scope:
            self._current_scope.dispose()
            self._current_scope = None
    
    def _create_instance(self, registration: ServiceRegistration) -> Any:
        """Create service instance with dependency injection
        
        Args:
            registration: Service registration
            
        Returns:
            Created instance
        """
        implementation = registration.implementation
        
        # Handle factory functions
        if callable(implementation) and not inspect.isclass(implementation):
            return implementation()
        
        # Handle classes with constructor injection
        if inspect.isclass(implementation):
            return self._create_class_instance(implementation)
        
        raise TypeError(f"Invalid implementation type: {type(implementation)}")
    
    def _create_class_instance(self, implementation_type: Type) -> Any:
        """Create class instance with dependency injection
        
        Args:
            implementation_type: Class to instantiate
            
        Returns:
            Class instance with injected dependencies
        """
        # Get constructor signature
        try:
            sig = inspect.signature(implementation_type.__init__)
        except (ValueError, TypeError):
            # No constructor or unusual constructor
            return implementation_type()
        
        # Get type hints for better dependency resolution
        try:
            type_hints = get_type_hints(implementation_type.__init__)
        except (NameError, AttributeError):
            type_hints = {}
        
        # Resolve constructor dependencies
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            # Try to get type from annotation or type hints
            param_type = None
            if param.annotation != param.empty:
                param_type = param.annotation
            elif param_name in type_hints:
                param_type = type_hints[param_name]
            
            if param_type is not None:
                try:
                    # Try to resolve dependency
                    dependency = self.resolve(param_type)
                    kwargs[param_name] = dependency
                except (ValueError, TypeError):
                    # Dependency not registered
                    if param.default == param.empty:
                        # Required parameter without default
                        raise ValueError(f"Cannot resolve required dependency {param_name} of type {param_type}")
                    # Use default value
                    continue
            elif param.default == param.empty:
                # Required parameter without type annotation
                raise ValueError(f"Cannot resolve parameter {param_name} without type annotation")
        
        return implementation_type(**kwargs)
    
    def dispose(self) -> None:
        """Dispose container and all managed instances"""
        # Dispose singletons
        for instance in self._singletons.values():
            if hasattr(instance, 'dispose'):
                instance.dispose()
        
        # Clear scope
        if self._current_scope:
            self._current_scope.dispose()
        
        self._services.clear()
        self._singletons.clear()
        self._current_scope = None

# Configuration helpers
def configure_container(config_path: Optional[Path] = None) -> Container:
    """Configure dependency injection container with default services
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured container
    """
    container = Container()
    
    # Register infrastructure services
    _register_infrastructure_services(container)
    
    # Register domain services
    _register_domain_services(container)
    
    # Register application services
    _register_application_services(container)
    
    # Load configuration if provided
    if config_path and config_path.exists():
        _load_configuration(container, config_path)
    
    return container

def configure_test_container() -> Container:
    """Configure container with mock implementations for testing
    
    Returns:
        Container with mock services
    """
    container = Container()
    
    # Register mock implementations
    _register_mock_services(container)
    
    return container

def _register_infrastructure_services(container: Container) -> None:
    """Register infrastructure layer services"""
    # Import here to avoid circular dependencies
    from .adapters.webcam_adapter import OpenCVWebcamAdapter
    from .adapters.cvat_adapter import CVATRestAdapter
    from .adapters.deepchecks_adapter import DeepChecksVisionAdapter
    from .adapters.ultralytics_adapter import UltralyticsYOLOAdapter
    from .infrastructure.file_storage import LocalFileStorage
    from .infrastructure.event_store import SQLiteEventStore
    from .infrastructure.event_bus import AsyncEventBus
    
    from .interfaces.data_capture import IWebcamDriver
    from .interfaces.annotation import ICVATAdapter
    from .interfaces.validation import IDeepChecksAdapter
    from .interfaces.training import IUltralyticsAdapter
    from .interfaces.events import IEventStore, IEventBus
    
    # Register adapters
    container.register_transient(IWebcamDriver, OpenCVWebcamAdapter)
    container.register_transient(ICVATAdapter, CVATRestAdapter)
    container.register_transient(IDeepChecksAdapter, DeepChecksVisionAdapter)
    container.register_transient(IUltralyticsAdapter, UltralyticsYOLOAdapter)
    
    # Register infrastructure services as singletons
    container.register_singleton(IEventStore, SQLiteEventStore)
    container.register_singleton(IEventBus, AsyncEventBus)

def _register_domain_services(container: Container) -> None:
    """Register domain layer services"""
    from .services.data_capture_service import DataCaptureService
    from .services.annotation_service import AnnotationService
    from .services.validation_service import ValidationService
    from .services.training_service import TrainingService
    
    from .interfaces.data_capture import IDataCapture
    from .interfaces.annotation import IAnnotationService
    from .interfaces.validation import IDataValidator, IModelValidator
    from .interfaces.training import IModelTrainer
    
    # Register domain services
    container.register_transient(IDataCapture, DataCaptureService)
    container.register_transient(IAnnotationService, AnnotationService)
    container.register_transient(IDataValidator, ValidationService)
    container.register_transient(IModelValidator, ValidationService)
    container.register_transient(IModelTrainer, TrainingService)

def _register_application_services(container: Container) -> None:
    """Register application layer services"""
    from .services.pipeline_orchestrator import PipelineOrchestrator
    from .services.configuration_service import ConfigurationService
    
    # Register application services
    container.register_transient(PipelineOrchestrator, PipelineOrchestrator)
    container.register_singleton(ConfigurationService, ConfigurationService)

def _register_mock_services(container: Container) -> None:
    """Register mock implementations for testing"""
    from .mocks.mock_webcam_driver import MockWebcamDriver
    from .mocks.mock_cvat_adapter import MockCVATAdapter
    from .mocks.mock_deepchecks_adapter import MockDeepChecksAdapter
    from .mocks.mock_ultralytics_adapter import MockUltralyticsAdapter
    from .mocks.mock_event_bus import MockEventBus
    from .mocks.mock_event_store import MockEventStore
    
    from .interfaces.data_capture import IWebcamDriver
    from .interfaces.annotation import ICVATAdapter
    from .interfaces.validation import IDeepChecksAdapter
    from .interfaces.training import IUltralyticsAdapter
    from .interfaces.events import IEventBus, IEventStore
    
    # Register mocks
    container.register_singleton(IWebcamDriver, MockWebcamDriver)
    container.register_singleton(ICVATAdapter, MockCVATAdapter)
    container.register_singleton(IDeepChecksAdapter, MockDeepChecksAdapter)
    container.register_singleton(IUltralyticsAdapter, MockUltralyticsAdapter)
    container.register_singleton(IEventBus, MockEventBus)
    container.register_singleton(IEventStore, MockEventStore)
    
    # Register domain services with mock dependencies
    from .services.data_capture_service import DataCaptureService
    from .services.annotation_service import AnnotationService
    from .services.validation_service import ValidationService
    from .services.training_service import TrainingService
    
    from .interfaces.data_capture import IDataCapture
    from .interfaces.annotation import IAnnotationService
    from .interfaces.validation import IDataValidator, IModelValidator
    from .interfaces.training import IModelTrainer
    
    container.register_transient(IDataCapture, DataCaptureService)
    container.register_transient(IAnnotationService, AnnotationService)
    container.register_transient(IDataValidator, ValidationService)
    container.register_transient(IModelValidator, ValidationService)
    container.register_transient(IModelTrainer, TrainingService)

def _load_configuration(container: Container, config_path: Path) -> None:
    """Load configuration from file"""
    # TODO: Implement configuration loading
    # This would read YAML/JSON config and override registrations
    pass

# Context manager for scoped services
class ServiceScopeContext:
    """Context manager for service scopes"""
    
    def __init__(self, container: Container):
        self.container = container
        self.scope = container.create_scope()
    
    def __enter__(self) -> ServiceScope:
        self.container.enter_scope(self.scope)
        return self.scope
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container.exit_scope()

# Container instance management
_default_container: Optional[Container] = None

def get_container() -> Container:
    """Get default container instance
    
    Returns:
        Default container instance
    """
    global _default_container
    if _default_container is None:
        _default_container = configure_container()
    return _default_container

def set_container(container: Container) -> None:
    """Set default container instance
    
    Args:
        container: Container to set as default
    """
    global _default_container
    _default_container = container

def reset_container() -> None:
    """Reset default container"""
    global _default_container
    if _default_container:
        _default_container.dispose()
        _default_container = None