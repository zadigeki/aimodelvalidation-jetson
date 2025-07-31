"""Event system interfaces for pipeline coordination"""

from typing import Protocol, Dict, Any, Callable, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import asyncio

class EventType(Enum):
    """Types of events in the ML pipeline"""
    
    # Pipeline events
    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"
    PIPELINE_CANCELLED = "pipeline.cancelled"
    
    # Stage events
    STAGE_STARTED = "stage.started"
    STAGE_COMPLETED = "stage.completed"
    STAGE_FAILED = "stage.failed"
    STAGE_SKIPPED = "stage.skipped"
    
    # Data capture events
    DATA_CAPTURE_STARTED = "data_capture.started"
    DATA_CAPTURED = "data.captured"
    DATA_CAPTURE_FAILED = "data_capture.failed"
    CAMERA_CONNECTED = "camera.connected"
    CAMERA_DISCONNECTED = "camera.disconnected"
    
    # Annotation events
    ANNOTATION_TASK_CREATED = "annotation.task_created"
    ANNOTATION_STARTED = "annotation.started"
    ANNOTATION_PROGRESS = "annotation.progress"
    ANNOTATIONS_COMPLETED = "annotations.completed"
    ANNOTATION_EXPORTED = "annotation.exported"
    ANNOTATION_FAILED = "annotation.failed"
    
    # Validation events
    VALIDATION_STARTED = "validation.started"
    VALIDATION_COMPLETED = "validation.completed"
    VALIDATION_FAILED = "validation.failed"
    VALIDATION_ISSUE_FOUND = "validation.issue_found"
    VALIDATION_THRESHOLD_EXCEEDED = "validation.threshold_exceeded"
    
    # Training events
    TRAINING_STARTED = "training.started"
    TRAINING_PROGRESS = "training.progress"
    TRAINING_EPOCH_COMPLETED = "training.epoch_completed"
    TRAINING_COMPLETED = "training.completed"
    TRAINING_FAILED = "training.failed"
    TRAINING_CANCELLED = "training.cancelled"
    TRAINING_CHECKPOINT_SAVED = "training.checkpoint_saved"
    
    # Model events
    MODEL_EXPORTED = "model.exported"
    MODEL_VALIDATED = "model.validated"
    MODEL_DEPLOYED = "model.deployed"
    
    # System events
    SYSTEM_ERROR = "system.error"
    RESOURCE_WARNING = "resource.warning"
    CONFIGURATION_CHANGED = "configuration.changed"

class EventPriority(Enum):
    """Event priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Event:
    """Event data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str
    source: str
    priority: EventPriority = EventPriority.NORMAL
    
    # Event metadata
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Retry information
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "id": self.id,
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "source": self.source,
            "priority": self.priority.value,
            "version": self.version,
            "tags": self.tags,
            "metadata": self.metadata,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        return cls(
            id=data["id"],
            type=EventType(data["type"]),
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data["correlation_id"],
            source=data["source"],
            priority=EventPriority(data.get("priority", "normal")),
            version=data.get("version", "1.0"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3)
        )

@dataclass
class EventFilter:
    """Filter for event subscriptions"""
    event_types: Optional[List[EventType]] = None
    sources: Optional[List[str]] = None
    correlation_ids: Optional[List[str]] = None
    priorities: Optional[List[EventPriority]] = None
    tags: Optional[List[str]] = None
    
    def matches(self, event: Event) -> bool:
        """Check if event matches filter"""
        if self.event_types and event.type not in self.event_types:
            return False
        if self.sources and event.source not in self.sources:
            return False
        if self.correlation_ids and event.correlation_id not in self.correlation_ids:
            return False
        if self.priorities and event.priority not in self.priorities:
            return False
        if self.tags and not any(tag in event.tags for tag in self.tags):
            return False
        return True

# Callback types
EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Union[None, asyncio.Task]]

@dataclass
class EventSubscription:
    """Event subscription information"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    handler: Union[EventHandler, AsyncEventHandler]
    filter: Optional[EventFilter] = None
    created_at: datetime = field(default_factory=datetime.now)
    active: bool = True

class IEventBus(Protocol):
    """Interface for event-driven communication"""
    
    async def publish(self, event: Event) -> None:
        """Publish event to bus
        
        Args:
            event: Event to publish
            
        Raises:
            EventError: If publishing fails
        """
        ...
    
    async def publish_batch(self, events: List[Event]) -> None:
        """Publish multiple events in batch
        
        Args:
            events: List of events to publish
        """
        ...
    
    async def subscribe(self, handler: Union[EventHandler, AsyncEventHandler], filter: Optional[EventFilter] = None) -> str:
        """Subscribe to events
        
        Args:
            handler: Event handler function
            filter: Optional event filter
            
        Returns:
            Subscription ID
        """
        ...
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events
        
        Args:
            subscription_id: Subscription identifier
            
        Returns:
            True if unsubscription successful
        """
        ...
    
    async def get_subscriptions(self) -> List[EventSubscription]:
        """Get all active subscriptions
        
        Returns:
            List of active subscriptions
        """
        ...
    
    async def start(self) -> None:
        """Start event bus"""
        ...
    
    async def stop(self) -> None:
        """Stop event bus"""
        ...
    
    async def health_check(self) -> bool:
        """Check event bus health
        
        Returns:
            True if healthy
        """
        ...

class IEventStore(Protocol):
    """Interface for event persistence and replay"""
    
    async def store_event(self, event: Event) -> None:
        """Store event for persistence
        
        Args:
            event: Event to store
            
        Raises:
            EventStoreError: If storage fails
        """
        ...
    
    async def store_batch(self, events: List[Event]) -> None:
        """Store multiple events in batch
        
        Args:
            events: List of events to store
        """
        ...
    
    async def get_events(
        self, 
        correlation_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Event]:
        """Retrieve events with filtering
        
        Args:
            correlation_id: Filter by correlation ID
            event_types: Filter by event types
            from_timestamp: Filter events after timestamp
            to_timestamp: Filter events before timestamp
            limit: Maximum number of events to return
            offset: Number of events to skip
            
        Returns:
            List of matching events
        """
        ...
    
    async def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Get specific event by ID
        
        Args:
            event_id: Event identifier
            
        Returns:
            Event if found, None otherwise
        """
        ...
    
    async def replay_events(
        self, 
        from_timestamp: datetime,
        to_timestamp: Optional[datetime] = None,
        event_types: Optional[List[EventType]] = None
    ) -> List[Event]:
        """Replay events from timestamp
        
        Args:
            from_timestamp: Start timestamp for replay
            to_timestamp: End timestamp for replay
            event_types: Filter by event types
            
        Returns:
            List of events for replay
        """
        ...
    
    async def delete_events(
        self,
        older_than: datetime,
        event_types: Optional[List[EventType]] = None
    ) -> int:
        """Delete old events
        
        Args:
            older_than: Delete events older than this timestamp
            event_types: Optional filter by event types
            
        Returns:
            Number of events deleted
        """
        ...
    
    async def get_event_stats(self) -> Dict[str, Any]:
        """Get event store statistics
        
        Returns:
            Statistics about stored events
        """
        ...

class IEventProcessor(Protocol):
    """Interface for processing events with business logic"""
    
    async def process_event(self, event: Event) -> Optional[List[Event]]:
        """Process event and optionally generate new events
        
        Args:
            event: Event to process
            
        Returns:
            Optional list of new events to publish
        """
        ...
    
    async def process_batch(self, events: List[Event]) -> List[Event]:
        """Process multiple events in batch
        
        Args:
            events: List of events to process
            
        Returns:
            List of new events to publish
        """
        ...
    
    def can_process(self, event: Event) -> bool:
        """Check if processor can handle event
        
        Args:
            event: Event to check
            
        Returns:
            True if processor can handle the event
        """
        ...

class IEventAggregator(Protocol):
    """Interface for aggregating related events"""
    
    async def aggregate_events(
        self,
        correlation_id: str,
        event_types: List[EventType],
        time_window: int = 300  # seconds
    ) -> Dict[str, Any]:
        """Aggregate events for analysis
        
        Args:
            correlation_id: Events to aggregate
            event_types: Types of events to include
            time_window: Time window for aggregation in seconds
            
        Returns:
            Aggregated event data
        """
        ...
    
    async def get_pipeline_summary(self, correlation_id: str) -> Dict[str, Any]:
        """Get summary of pipeline execution
        
        Args:
            correlation_id: Pipeline correlation ID
            
        Returns:
            Pipeline execution summary
        """
        ...
    
    async def get_performance_metrics(
        self,
        from_timestamp: datetime,
        to_timestamp: datetime
    ) -> Dict[str, Any]:
        """Get performance metrics for time period
        
        Args:
            from_timestamp: Start of time period
            to_timestamp: End of time period
            
        Returns:
            Performance metrics
        """
        ...

# Event creation helpers
class EventFactory:
    """Factory for creating common events"""
    
    @staticmethod
    def create_pipeline_started(correlation_id: str, config: Dict[str, Any]) -> Event:
        """Create pipeline started event"""
        return Event(
            type=EventType.PIPELINE_STARTED,
            payload={"config": config},
            correlation_id=correlation_id,
            source="pipeline_orchestrator",
            priority=EventPriority.HIGH
        )
    
    @staticmethod
    def create_stage_completed(correlation_id: str, stage: str, result: Dict[str, Any]) -> Event:
        """Create stage completed event"""
        return Event(
            type=EventType.STAGE_COMPLETED,
            payload={"stage": stage, "result": result},
            correlation_id=correlation_id,
            source="pipeline_orchestrator",
            tags=[stage]
        )
    
    @staticmethod
    def create_training_progress(correlation_id: str, progress: Dict[str, Any]) -> Event:
        """Create training progress event"""
        return Event(
            type=EventType.TRAINING_PROGRESS,
            payload=progress,
            correlation_id=correlation_id,
            source="model_trainer",
            tags=["training", "progress"]
        )
    
    @staticmethod
    def create_validation_issue(correlation_id: str, issue: Dict[str, Any]) -> Event:
        """Create validation issue event"""
        priority = EventPriority.CRITICAL if issue.get("severity") == "critical" else EventPriority.HIGH
        return Event(
            type=EventType.VALIDATION_ISSUE_FOUND,
            payload=issue,
            correlation_id=correlation_id,
            source="data_validator",
            priority=priority,
            tags=["validation", "issue", issue.get("severity", "unknown")]
        )
    
    @staticmethod
    def create_error_event(correlation_id: str, error: Exception, source: str) -> Event:
        """Create system error event"""
        return Event(
            type=EventType.SYSTEM_ERROR,
            payload={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_details": getattr(error, 'details', {})
            },
            correlation_id=correlation_id,
            source=source,
            priority=EventPriority.CRITICAL,
            tags=["error", "system"]
        )

# Exceptions
class EventError(Exception):
    """Base exception for event system"""
    
    def __init__(self, message: str, error_code: str = "EVENT_ERROR", details: Dict[str, Any] | None = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

class EventPublishError(EventError):
    """Exception when event publishing fails"""
    pass

class EventStoreError(EventError):
    """Exception when event storage fails"""
    pass

class EventSubscriptionError(EventError):
    """Exception when event subscription fails"""
    pass

class EventProcessingError(EventError):
    """Exception when event processing fails"""
    pass