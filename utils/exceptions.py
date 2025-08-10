"""
Custom exception classes for the Course Recommendation System.
Each exception provides context-specific error information for precise failure handling.
"""

class CourseRecommendationError(Exception):
    """Base exception class for all course recommendation system errors."""
    pass


class DatabaseConnectionError(CourseRecommendationError):
    """Raised when database connections (MySQL or Neo4j) fail."""
    pass


class DatabaseQueryError(CourseRecommendationError):
    """Raised when database queries fail to execute."""
    pass


class APIKeyError(CourseRecommendationError):
    """Raised when API keys are invalid or missing."""
    pass


class APIRequestError(CourseRecommendationError):
    """Raised when external API requests (Cohere, Tavily) fail."""
    pass


class VectorStoreError(CourseRecommendationError):
    """Raised when vector store operations (FAISS) fail."""
    pass


class FileProcessingError(CourseRecommendationError):
    """Raised when file processing (PDF, DOCX) fails."""
    pass


class ConfigurationError(CourseRecommendationError):
    """Raised when system configuration is invalid."""
    pass


class AgentExecutionError(CourseRecommendationError):
    """Raised when agent execution fails."""
    pass


class UserInputValidationError(CourseRecommendationError):
    """Raised when user input validation fails."""
    pass


class WorkflowOrchestrationError(CourseRecommendationError):
    """Raised when workflow orchestration fails."""
    pass


class WorkflowError(CourseRecommendationError):
    """Raised when workflow execution encounters errors."""
    pass