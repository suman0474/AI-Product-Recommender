"""
Planning data models for Level 4 autonomous execution.

This module defines the core data structures for goal decomposition,
task planning, and DAG-based execution.
"""

from typing import TypedDict, List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class GoalStatus(str, Enum):
    """Status of a goal in the planning system."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    """Status of a task in the execution DAG."""
    PENDING = "pending"
    READY = "ready"  # Dependencies satisfied, ready to execute
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class PlanValidationStatus(str, Enum):
    """Result of plan validation."""
    VALID = "valid"
    INVALID_CYCLE = "invalid_cycle"
    INVALID_UNREACHABLE = "invalid_unreachable"
    INVALID_MISSING_TOOL = "invalid_missing_tool"
    INVALID_CONSTRAINT = "invalid_constraint"


# ============================================================================
# TypedDict Definitions (for LangGraph State)
# ============================================================================

class Goal(TypedDict):
    """
    Represents a goal or sub-goal in the planning hierarchy.

    Goals are decomposed from user requests and form a tree structure
    where sub-goals contribute to parent goal completion.
    """
    goal_id: str
    description: str
    priority: int  # 1=highest, 10=lowest
    status: str  # GoalStatus value
    parent_goal_id: Optional[str]  # None for primary goal
    dependencies: List[str]  # Goal IDs this depends on
    success_criteria: str  # How to determine completion
    created_at: str
    updated_at: str
    completed_at: Optional[str]
    result: Optional[Dict[str, Any]]
    error: Optional[str]


class Task(TypedDict):
    """
    Represents an executable task in the DAG.

    Tasks are atomic units of work that map to tool invocations.
    They form the leaves of the planning tree.
    """
    task_id: str
    goal_id: str  # Which goal this task serves
    description: str
    tool_required: Optional[str]  # Tool name to execute
    tool_capability: Optional[str]  # Required capability if tool not specified
    input_params: Dict[str, Any]
    output_key: str  # Key in state to store result
    dependencies: List[str]  # Task IDs this depends on
    status: str  # TaskStatus value
    priority: int
    estimated_duration_ms: int
    actual_duration_ms: Optional[int]
    retry_count: int
    max_retries: int
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    result: Optional[Dict[str, Any]]
    error: Optional[str]


class ExecutionDAG(TypedDict):
    """
    Directed Acyclic Graph representing the execution plan.

    The DAG encodes task dependencies and enables parallel execution
    of independent tasks while respecting ordering constraints.
    """
    dag_id: str
    goals: List[Goal]
    tasks: List[Task]
    edges: List[Tuple[str, str]]  # (from_task_id, to_task_id)
    topological_order: List[str]  # Task IDs in execution order
    critical_path: List[str]  # Longest path through DAG
    estimated_total_duration_ms: int
    parallelism_factor: float  # Ratio of parallel to sequential execution
    created_at: str
    validation_status: str  # PlanValidationStatus value
    validation_errors: List[str]


class PlanningState(TypedDict):
    """
    State for the planning workflow.

    This is the main state object passed through the LangGraph
    planning workflow nodes.
    """
    session_id: str
    user_input: str
    domain_context: Optional[str]

    # Goal hierarchy
    primary_goal: Optional[Goal]
    sub_goals: List[Goal]

    # Execution plan
    execution_dag: Optional[ExecutionDAG]

    # Execution tracking
    current_task_id: Optional[str]
    completed_tasks: List[str]
    failed_tasks: List[str]
    skipped_tasks: List[str]

    # Results
    intermediate_results: Dict[str, Any]
    final_result: Optional[Dict[str, Any]]

    # Plan versioning (for replanning)
    plan_version: int
    replan_triggers: List[str]
    replan_count: int
    max_replans: int

    # Metadata
    created_at: str
    updated_at: str
    status: str  # Overall planning status
    error: Optional[str]


# ============================================================================
# Pydantic Models (for API and validation)
# ============================================================================

class GoalModel(BaseModel):
    """Pydantic model for Goal validation and serialization."""
    goal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str = Field(..., min_length=1, max_length=1000)
    priority: int = Field(default=5, ge=1, le=10)
    status: GoalStatus = Field(default=GoalStatus.PENDING)
    parent_goal_id: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    success_criteria: str = Field(default="Task completed successfully")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_typed_dict(self) -> Goal:
        """Convert to TypedDict for LangGraph state."""
        return Goal(
            goal_id=self.goal_id,
            description=self.description,
            priority=self.priority,
            status=self.status.value,
            parent_goal_id=self.parent_goal_id,
            dependencies=self.dependencies,
            success_criteria=self.success_criteria,
            created_at=self.created_at.isoformat(),
            updated_at=self.updated_at.isoformat(),
            completed_at=self.completed_at.isoformat() if self.completed_at else None,
            result=self.result,
            error=self.error
        )


class TaskModel(BaseModel):
    """Pydantic model for Task validation and serialization."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal_id: str
    description: str = Field(..., min_length=1, max_length=500)
    tool_required: Optional[str] = None
    tool_capability: Optional[str] = None
    input_params: Dict[str, Any] = Field(default_factory=dict)
    output_key: str = Field(default="result")
    dependencies: List[str] = Field(default_factory=list)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    priority: int = Field(default=5, ge=1, le=10)
    estimated_duration_ms: int = Field(default=5000, ge=0)
    actual_duration_ms: Optional[int] = None
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_typed_dict(self) -> Task:
        """Convert to TypedDict for LangGraph state."""
        return Task(
            task_id=self.task_id,
            goal_id=self.goal_id,
            description=self.description,
            tool_required=self.tool_required,
            tool_capability=self.tool_capability,
            input_params=self.input_params,
            output_key=self.output_key,
            dependencies=self.dependencies,
            status=self.status.value,
            priority=self.priority,
            estimated_duration_ms=self.estimated_duration_ms,
            actual_duration_ms=self.actual_duration_ms,
            retry_count=self.retry_count,
            max_retries=self.max_retries,
            created_at=self.created_at.isoformat(),
            started_at=self.started_at.isoformat() if self.started_at else None,
            completed_at=self.completed_at.isoformat() if self.completed_at else None,
            result=self.result,
            error=self.error
        )


class ExecutionDAGModel(BaseModel):
    """Pydantic model for ExecutionDAG validation."""
    dag_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goals: List[GoalModel] = Field(default_factory=list)
    tasks: List[TaskModel] = Field(default_factory=list)
    edges: List[Tuple[str, str]] = Field(default_factory=list)
    topological_order: List[str] = Field(default_factory=list)
    critical_path: List[str] = Field(default_factory=list)
    estimated_total_duration_ms: int = Field(default=0, ge=0)
    parallelism_factor: float = Field(default=1.0, ge=0.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    validation_status: PlanValidationStatus = Field(default=PlanValidationStatus.VALID)
    validation_errors: List[str] = Field(default_factory=list)


class PlanRequest(BaseModel):
    """API request model for creating a new plan."""
    user_input: str = Field(..., min_length=1, max_length=5000)
    domain_context: Optional[str] = Field(default=None, max_length=1000)
    session_id: Optional[str] = None
    max_replans: int = Field(default=3, ge=0, le=10)


class PlanResponse(BaseModel):
    """API response model for plan status."""
    success: bool
    dag_id: Optional[str] = None
    status: str
    primary_goal: Optional[GoalModel] = None
    sub_goals_count: int = 0
    tasks_count: int = 0
    estimated_duration_ms: int = 0
    error: Optional[str] = None


# ============================================================================
# Factory Functions
# ============================================================================

def create_planning_state(
    session_id: str,
    user_input: str,
    domain_context: Optional[str] = None,
    max_replans: int = 3
) -> PlanningState:
    """
    Factory function to create a new PlanningState.

    Args:
        session_id: Unique session identifier
        user_input: The user's request to plan for
        domain_context: Optional domain/industry context
        max_replans: Maximum number of replanning attempts

    Returns:
        Initialized PlanningState TypedDict
    """
    now = datetime.utcnow().isoformat()
    return PlanningState(
        session_id=session_id,
        user_input=user_input,
        domain_context=domain_context,
        primary_goal=None,
        sub_goals=[],
        execution_dag=None,
        current_task_id=None,
        completed_tasks=[],
        failed_tasks=[],
        skipped_tasks=[],
        intermediate_results={},
        final_result=None,
        plan_version=1,
        replan_triggers=[],
        replan_count=0,
        max_replans=max_replans,
        created_at=now,
        updated_at=now,
        status=GoalStatus.PENDING.value,
        error=None
    )


def create_goal(
    description: str,
    priority: int = 5,
    parent_goal_id: Optional[str] = None,
    dependencies: Optional[List[str]] = None,
    success_criteria: str = "Goal completed successfully"
) -> Goal:
    """
    Factory function to create a new Goal.

    Args:
        description: What the goal aims to achieve
        priority: Priority level (1=highest, 10=lowest)
        parent_goal_id: ID of parent goal if this is a sub-goal
        dependencies: List of goal IDs this depends on
        success_criteria: How to determine goal completion

    Returns:
        Initialized Goal TypedDict
    """
    now = datetime.utcnow().isoformat()
    return Goal(
        goal_id=str(uuid.uuid4()),
        description=description,
        priority=priority,
        status=GoalStatus.PENDING.value,
        parent_goal_id=parent_goal_id,
        dependencies=dependencies or [],
        success_criteria=success_criteria,
        created_at=now,
        updated_at=now,
        completed_at=None,
        result=None,
        error=None
    )


def create_task(
    goal_id: str,
    description: str,
    tool_required: Optional[str] = None,
    tool_capability: Optional[str] = None,
    input_params: Optional[Dict[str, Any]] = None,
    output_key: str = "result",
    dependencies: Optional[List[str]] = None,
    priority: int = 5,
    estimated_duration_ms: int = 5000,
    max_retries: int = 3
) -> Task:
    """
    Factory function to create a new Task.

    Args:
        goal_id: ID of the goal this task serves
        description: What the task does
        tool_required: Specific tool to use (optional)
        tool_capability: Required capability if tool not specified
        input_params: Parameters to pass to the tool
        output_key: Key to store result in state
        dependencies: List of task IDs this depends on
        priority: Execution priority
        estimated_duration_ms: Estimated execution time
        max_retries: Maximum retry attempts on failure

    Returns:
        Initialized Task TypedDict
    """
    now = datetime.utcnow().isoformat()
    return Task(
        task_id=str(uuid.uuid4()),
        goal_id=goal_id,
        description=description,
        tool_required=tool_required,
        tool_capability=tool_capability,
        input_params=input_params or {},
        output_key=output_key,
        dependencies=dependencies or [],
        status=TaskStatus.PENDING.value,
        priority=priority,
        estimated_duration_ms=estimated_duration_ms,
        actual_duration_ms=None,
        retry_count=0,
        max_retries=max_retries,
        created_at=now,
        started_at=None,
        completed_at=None,
        result=None,
        error=None
    )


def create_execution_dag(
    goals: List[Goal],
    tasks: List[Task],
    edges: List[Tuple[str, str]]
) -> ExecutionDAG:
    """
    Factory function to create a new ExecutionDAG.

    Args:
        goals: List of goals in the plan
        tasks: List of tasks to execute
        edges: Dependency edges between tasks

    Returns:
        Initialized ExecutionDAG TypedDict
    """
    now = datetime.utcnow().isoformat()

    # Calculate estimated total duration (sum of all tasks for now)
    total_duration = sum(t.get("estimated_duration_ms", 0) for t in tasks)

    return ExecutionDAG(
        dag_id=str(uuid.uuid4()),
        goals=goals,
        tasks=tasks,
        edges=edges,
        topological_order=[],  # Will be computed by dag_builder
        critical_path=[],  # Will be computed by dag_builder
        estimated_total_duration_ms=total_duration,
        parallelism_factor=1.0,  # Will be computed by dag_builder
        created_at=now,
        validation_status=PlanValidationStatus.VALID.value,
        validation_errors=[]
    )
