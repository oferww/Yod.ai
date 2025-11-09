"""
Metrics tracking utility using Langfuse for monitoring chatbot performance.

Tracks:
- Number of turns per session
- Latency (response time)
- User feedback
- LLM API calls and tokens
- Recommendations given
- Session metadata
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from functools import wraps

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks metrics for chatbot sessions using Langfuse."""
    
    def __init__(self, session_id: str, enabled: bool = True):
        """Initialize metrics tracker.
        
        Args:
            session_id: Unique session identifier
            enabled: Whether to track metrics (can be disabled if Langfuse not configured)
        """
        self.session_id = session_id
        self.enabled = enabled and LANGFUSE_AVAILABLE
        self.langfuse = None
        self.trace = None
        self.turn_count = 0
        self.start_time = datetime.now()
        self.session_metadata = {
            "session_id": session_id,
            "started_at": self.start_time.isoformat(),
            "turns": [],
            "total_turns": 0,
            "total_latency_ms": 0,
            "feedback": None,
            "recommendations_given": [],
            "llm_calls": 0,
            "total_tokens": {"input": 0, "output": 0},
        }
        
        if self.enabled:
            try:
                self.langfuse = Langfuse(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                )
                # Create a trace for this session
                self.trace = self.langfuse.trace(
                    name="chatbot_session",
                    session_id=session_id,
                    metadata={
                        "started_at": self.start_time.isoformat(),
                    }
                )
                print(f"Langfuse metrics tracking enabled for session {session_id}", flush=True)
            except Exception as e:
                print(f"Failed to initialize Langfuse: {e}. Metrics tracking disabled.", flush=True)
                self.enabled = False
    
    def track_turn(self, user_message: str, assistant_response: str, 
                   latency_ms: float, llm_metadata: Optional[Dict[str, Any]] = None,
                   start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                   intent: Optional[str] = None, response_type: Optional[str] = None,
                   products: Optional[list] = None, matched_count: int = 0, filter_type: Optional[str] = None):
        """Track a single conversation turn with all related data.
        
        Args:
            user_message: User's input message
            assistant_response: Assistant's response
            latency_ms: Response time in milliseconds (time to first chunk)
            llm_metadata: Optional metadata about LLM call (tokens, model, etc.)
            start_time: Optional start time (datetime). If provided with end_time, used for Langfuse latency tracking
            end_time: Optional end time (datetime). If provided with start_time, used for Langfuse latency tracking
            intent: Detected user intent
            response_type: Type of response (recommendation, greeting, etc.)
            products: List of recommended products (if any)
            matched_count: Number of products that matched criteria
            filter_type: Type of filtering used ('semantic', 'hard_specs', or 'hybrid')
        """
        self.turn_count += 1
        turn_data = {
            "turn_number": self.turn_count,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat(),
            "llm_metadata": llm_metadata or {},
        }
        
        self.session_metadata["turns"].append(turn_data)
        self.session_metadata["total_turns"] = self.turn_count
        self.session_metadata["total_latency_ms"] += latency_ms
        
        # Track LLM calls and tokens
        if llm_metadata:
            self.session_metadata["llm_calls"] += 1
            print(f"[METRICS_TOKENS] Processing llm_metadata: {llm_metadata}", flush=True)
            if "input_tokens" in llm_metadata:
                input_tokens = llm_metadata["input_tokens"]
                self.session_metadata["total_tokens"]["input"] += input_tokens
                print(f"[METRICS_TOKENS] ✓ Added input_tokens: {input_tokens}, total input: {self.session_metadata['total_tokens']['input']}", flush=True)
            if "output_tokens" in llm_metadata:
                output_tokens = llm_metadata["output_tokens"]
                self.session_metadata["total_tokens"]["output"] += output_tokens
                print(f"[METRICS_TOKENS] ✓ Added output_tokens: {output_tokens}, total output: {self.session_metadata['total_tokens']['output']}", flush=True)
        
        if self.enabled and self.trace:
            try:
                # Build comprehensive metadata
                metadata = {
                    "turn_number": self.turn_count,
                    "latency_ms": latency_ms,
                }
                
                # Add intent and response type
                if intent:
                    metadata["intent"] = intent
                if response_type:
                    metadata["response_type"] = response_type
                if filter_type:
                    metadata["filter_type"] = filter_type
                
                # Add product information
                if products is not None:
                    metadata["products_recommended"] = len(products)
                    metadata["matched_count"] = matched_count
                    if products:
                        # Store each SKU as individual field for Langfuse dashboard filtering
                        skus = [p.get("sku") for p in products]
                        metadata["product_skus"] = ", ".join(skus)  # Comma-separated string for dashboard
                        # Also add individual SKU fields for filtering
                        for idx, sku in enumerate(skus):
                            metadata[f"product_sku_{idx}"] = sku
                
                # Add LLM metadata (tokens, model, etc.)
                if llm_metadata:
                    if "input_tokens" in llm_metadata:
                        metadata["prompt_tokens"] = llm_metadata["input_tokens"]
                    if "output_tokens" in llm_metadata:
                        metadata["response_tokens"] = llm_metadata["output_tokens"]
                    if "model" in llm_metadata:
                        metadata["llm_model"] = llm_metadata["model"]
                    if "provider" in llm_metadata:
                        metadata["llm_provider"] = llm_metadata["provider"]
                    # Include any other metadata
                    for key, value in llm_metadata.items():
                        if key not in ["input_tokens", "output_tokens", "model", "provider"]:
                            metadata[key] = value
                
                # Create Generation observation (for LLM calls)
                generation_kwargs = {
                    "name": f"turn_{self.turn_count}",
                    "model": llm_metadata.get("model") if llm_metadata else None,
                    "input": user_message,
                    "output": assistant_response,
                    "metadata": metadata
                }
                
                # Add model parameters if available
                if llm_metadata:
                    if "provider" in llm_metadata:
                        generation_kwargs["modelParameters"] = {
                            "provider": llm_metadata["provider"]
                        }
                
                # Add usage (token counts) if available
                if llm_metadata:
                    usage = {}
                    if "input_tokens" in llm_metadata:
                        usage["input"] = llm_metadata["input_tokens"]
                    if "output_tokens" in llm_metadata:
                        usage["output"] = llm_metadata["output_tokens"]
                    if usage:
                        generation_kwargs["usage"] = usage
                        print(f"[METRICS_LANGFUSE] ✓ Added usage to Langfuse generation: {usage}", flush=True)
                
                # Add startTime and endTime if provided for built-in latency tracking
                if start_time and end_time:
                    # Langfuse expects camelCase parameter names
                    generation_kwargs["startTime"] = start_time
                    generation_kwargs["endTime"] = end_time
                    # Debug: log the times being sent to Langfuse
                    duration_ms = (end_time - start_time).total_seconds() * 1000
                    print(f"Turn {self.turn_count} - Langfuse times: start={start_time.isoformat()}, end={end_time.isoformat()}, duration={duration_ms:.2f}ms", flush=True)
                else:
                    print(f"Turn {self.turn_count} - No start_time or end_time provided for Langfuse tracking", flush=True)
                
                print(f"Creating Langfuse generation with metadata: {list(metadata.keys())}", flush=True)
                print(f"[METRICS_LANGFUSE] Creating generation observation for turn {self.turn_count}", flush=True)
                print(f"[METRICS_LANGFUSE] Generation kwargs keys: {list(generation_kwargs.keys())}", flush=True)
                if "usage" in generation_kwargs:
                    print(f"[METRICS_LANGFUSE] ✓ Generation includes usage data: {generation_kwargs['usage']}", flush=True)
                # Log API call counts if present
                if "total_api_calls" in metadata:
                    print(f"[METRICS_LANGFUSE] ✓ Generation includes API metrics: total_api_calls={metadata['total_api_calls']}, breakdown={metadata.get('api_calls_breakdown', {})}", flush=True)
                self.trace.generation(**generation_kwargs)
                print(f"[METRICS_LANGFUSE] ✓ Generation observation created successfully", flush=True)
            except Exception as e:
                print(f"Failed to track turn in Langfuse: {e}", flush=True)
    
    def track_recommendation(self, products: list, matched_count: int, 
                            filter_type: str = "hybrid",
                            start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                            response_type: Optional[str] = None):
        """DEPRECATED: Use track_turn() with products parameter instead.
        
        This method is kept for backward compatibility but does nothing.
        All tracking is now done in track_turn() as a single observation.
        
        Args:
            products: List of recommended products (empty if no recommendation)
            matched_count: Number of products that matched criteria
            filter_type: Type of filtering used (hybrid, semantic, spec-based)
            start_time: Optional start time (datetime) for latency tracking
            end_time: Optional end time (datetime) for latency tracking
            response_type: Type of response (recommendation, clarification, greeting, etc.)
        """
        # Only log to local session metadata, no Langfuse span
        # (Langfuse tracking is now done in track_turn())
        recommendation_data = {
            "timestamp": datetime.now().isoformat(),
            "products_recommended": len(products),
            "matched_count": matched_count,
            "filter_type": filter_type,
            "product_skus": [p.get("sku") for p in products] if products else [],
        }
        
        self.session_metadata["recommendations_given"].append(recommendation_data)
        print(f"Logged recommendation data locally (no separate Langfuse span)", flush=True)
    
    def track_feedback(self, feedback: str, rating: Optional[int] = None):
        """Track user feedback on the recommendation.
        
        Args:
            feedback: User's feedback text
            rating: Optional rating (e.g., 1-5 stars)
        """
        self.session_metadata["feedback"] = {
            "text": feedback,
            "rating": rating,
            "timestamp": datetime.now().isoformat(),
        }
        
        if self.enabled and self.trace:
            try:
                self.trace.span(
                    name="user_feedback",
                    input={
                        "feedback": feedback,
                        "rating": rating,
                    }
                )
            except Exception as e:
                print(f"Failed to track feedback in Langfuse: {e}", flush=True)
    
    def track_error(self, error_message: str, error_type: str = "general",
                    start_time: Optional[datetime] = None, end_time: Optional[datetime] = None):
        """Track an error that occurred during the session.
        
        Args:
            error_message: Description of the error
            error_type: Type of error (e.g., "llm_error", "filter_error")
            start_time: Optional start time (datetime) for latency tracking
            end_time: Optional end time (datetime) for latency tracking
        """
        if self.enabled and self.trace:
            try:
                span_kwargs = {
                    "name": "error",
                    "input": {
                        "error_type": error_type,
                        "error_message": error_message,
                    },
                    "level": "ERROR"
                }
                
                # Add startTime and endTime if provided for built-in latency tracking
                if start_time and end_time:
                    # Langfuse expects camelCase parameter names
                    span_kwargs["startTime"] = start_time
                    span_kwargs["endTime"] = end_time
                
                self.trace.span(**span_kwargs)
            except Exception as e:
                print(f"Failed to track error in Langfuse: {e}", flush=True)
        
        print(f"Session {self.session_id} - {error_type}: {error_message}", flush=True)
    
    def end_session(self):
        """End the session and flush metrics to Langfuse."""
        end_time = datetime.now()
        duration_seconds = (end_time - self.start_time).total_seconds()
        
        self.session_metadata["ended_at"] = end_time.isoformat()
        self.session_metadata["duration_seconds"] = duration_seconds
        self.session_metadata["average_latency_ms"] = (
            self.session_metadata["total_latency_ms"] / self.turn_count 
            if self.turn_count > 0 else 0
        )
        
        if self.enabled and self.trace:
            try:
                # Use Langfuse's built-in latency tracking with startTime and endTime
                # Langfuse expects camelCase parameter names
                self.trace.update(
                    output=self.session_metadata,
                    startTime=self.start_time,
                    endTime=end_time,
                    metadata={
                        "total_turns": self.turn_count,
                        "average_latency_ms": self.session_metadata["average_latency_ms"],
                    }
                )
                self.langfuse.flush()
                print(f"Session {self.session_id} metrics flushed to Langfuse", flush=True)
            except Exception as e:
                print(f"Failed to end session in Langfuse: {e}", flush=True)
    
    def get_session_metrics(self) -> Dict[str, Any]:
        """Get current session metrics.
        
        Returns:
            Dictionary containing all session metrics
        """
        return self.session_metadata.copy()


def get_metrics_tracker(session_id: str) -> MetricsTracker:
    """Factory function to create a metrics tracker.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        MetricsTracker instance
    """
    # Check if Langfuse is configured
    enabled = bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and 
        os.getenv("LANGFUSE_SECRET_KEY")
    )
    
    if not enabled:
        print("Langfuse not configured. Metrics will be tracked locally only.", flush=True)
    
    return MetricsTracker(session_id, enabled=enabled)
