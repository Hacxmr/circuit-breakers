"""
NBF-LLM Integration with Circuit Breaker

This module integrates the Neural Barrier Function (NBF) approach from
"Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks"
(Hu, Robey, & Liu) with our context-aware circuit breaker.

Key innovations:
1. Dynamic threshold (η) modulation based on conversation context
2. Cross-architecture generalization through universal embeddings
3. Online adversarial training loop
4. Circuit breaker triggers when NBF detects safety violations
"""

import sys
from pathlib import Path
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'nbf_llm'))

try:
    from circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerOpenError
    )
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    print("Warning: Circuit breaker modules not available. Using placeholder.")
    CircuitBreaker = None
    CircuitBreakerConfig = None
    CircuitBreakerOpenError = Exception
    CIRCUIT_BREAKER_AVAILABLE = False


@dataclass
class NBFConfig:
    """Configuration for NBF-based detection"""
    model_path: str = "nbf_llm/models/models_best_nbf_released.pth"
    base_eta: float = 0.001  # Base threshold
    min_eta: float = 0.0001  # Minimum threshold (most permissive)
    max_eta: float = 0.01    # Maximum threshold (most restrictive)
    
    # Context-based modulation factors
    creative_reduction: float = 0.5  # Reduce eta for creative tasks
    technical_reduction: float = 0.3  # Reduce eta for technical help
    high_stakes_increase: float = 3.0  # Increase eta for sensitive topics
    escalation_increase: float = 2.0  # Increase eta on escalation
    
    # Safety invariance parameters
    window_size: int = 3  # Number of recent turns to analyze
    safety_decay: float = 0.9  # How quickly safety scores decay
    
    # Semantic classifier settings
    use_semantic_classifier: bool = False  # Use pretrained model instead of keywords
    semantic_model_name: str = "unitary/toxic-bert"  # HuggingFace model
    semantic_threshold: float = 0.7  # Confidence threshold for harmful classification
    fallback_to_keywords: bool = True  # Use keywords if semantic model unavailable
    
    # Semantic classifier settings
    use_semantic_classifier: bool = False  # Use pretrained model instead of keywords
    semantic_model_name: str = "unitary/toxic-bert"  # HuggingFace model
    semantic_threshold: float = 0.7  # Confidence threshold for harmful classification
    fallback_to_keywords: bool = True  # Use keywords if semantic model unavailable


@dataclass
class SafetyState:
    """Represents the current safety state of conversation"""
    safety_index: float  # NBF safety score
    trajectory: List[float]  # Historical safety scores
    steering_active: bool  # Whether steering is currently applied
    circuit_breaker_triggered: bool  # Whether circuit breaker has opened
    eta: float  # Current dynamic threshold
    turn_count: int  # Number of turns in conversation


class DynamicNBFIntegration:
    """
    Integrates NBF-LLM with circuit breaker using dynamic threshold modulation.
    
    This addresses the paper's limitation: "A fixed steering threshold η 
    leads to over-refusal." We make η context-aware and adaptive.
    """
    
    def __init__(self, 
                 nbf_config: Optional[NBFConfig] = None,
                 use_circuit_breaker: bool = True):
        """
        Initialize the integrated system.
        
        Args:
            nbf_config: NBF configuration
            use_circuit_breaker: Whether to enable circuit breaker
        """
        self.config = nbf_config or NBFConfig()
        self.use_circuit_breaker = use_circuit_breaker
        
        # Load NBF models
        self.nbf_models = self._load_nbf_models()
        
        # Initialize circuit breaker if available
        self.circuit_breaker = None
        if use_circuit_breaker and CIRCUIT_BREAKER_AVAILABLE:
            cb_config = CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=2,
                timeout=60.0,
                failure_rate_threshold=0.5
            )
            self.circuit_breaker = CircuitBreaker(config=cb_config)
        
        # Track safety state
        self.safety_state = SafetyState(
            safety_index=0.0,
            trajectory=[],
            steering_active=False,
            circuit_breaker_triggered=False,
            eta=self.config.base_eta,
            turn_count=0
        )
        
        # Load semantic classifier if enabled
        self.semantic_classifier = None
        self.semantic_tokenizer = None
        if self.config.use_semantic_classifier:
            try:
                self.semantic_classifier, self.semantic_tokenizer = self._load_semantic_classifier()
            except Exception as e:
                print(f"Warning: Failed to load semantic classifier: {e}")
                if not self.config.fallback_to_keywords:
                    raise
        
        # Context classification cache
        self.conversation_context = "general"
    
    def _load_nbf_models(self) -> Optional[Dict]:
        """Load pre-trained NBF models"""
        try:
            if Path(self.config.model_path).exists():
                models = torch.load(
                    self.config.model_path, 
                    map_location=torch.device('cpu')
                )
                print(f"Loaded NBF models from {self.config.model_path}")
                return models
            else:
                print(f"Warning: NBF model not found at {self.config.model_path}")
                print("Operating in simulation mode without actual NBF steering.")
                return None
        except Exception as e:
            print(f"Error loading NBF models: {e}")
            return None
    
    def _load_semantic_classifier(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Load pretrained semantic classifier for content safety."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            print(f"Loading semantic classifier: {self.config.semantic_model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(self.config.semantic_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.semantic_model_name
            )
            model.eval()
            
            print(f"Semantic classifier loaded successfully")
            return model, tokenizer
            
        except ImportError:
            print("Warning: transformers library not available")
            if self.config.fallback_to_keywords:
                print("Falling back to keywords")
            return None, None
        except Exception as e:
            print(f"Warning: Could not load semantic classifier: {e}")
            if self.config.fallback_to_keywords:
                print("Falling back to keywords")
            return None, None
    
    def _classify_with_semantic_model(self, text: str) -> Optional[str]:
        """Classify text using semantic model with instruction pattern detection."""
        if self.semantic_classifier is None or self.semantic_tokenizer is None:
            return None
        
        try:
            inputs = self.semantic_tokenizer(
                text, return_tensors="pt", truncation=True, 
                max_length=512, padding=True
            )
            
            with torch.no_grad():
                outputs = self.semantic_classifier(**inputs)
                probs = torch.sigmoid(outputs.logits)[0]
            
            labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            max_prob = probs.max().item()
            max_idx = probs.argmax().item()
            max_label = labels[max_idx] if max_idx < len(labels) else 'unknown'
            
            # Check for dangerous instruction patterns
            text_lower = text.lower()
            instruction_patterns = ['how to make', 'how to create', 'how to build', 'how do i make',
                                   'instructions for', 'steps to', 'guide to', 'tutorial on',
                                   'methods for making', 'ways to make', 'process of making']
            dangerous_keywords = ['bomb', 'explosive', 'explosives', 'weapon', 'poison', 'drug', 
                                'methamphetamine', 'meth', 'cocaine', 'heroin', 'fentanyl', 
                                'detonator', 'grenade', 'ammunition', 'c4', 'tnt', 'dynamite',
                                'amphetamine', 'mdma', 'lsd', 'narcotic']
            
            has_instruction = any(pattern in text_lower for pattern in instruction_patterns)
            has_dangerous = any(keyword in text_lower for keyword in dangerous_keywords)
            
            # If it's a "how to make" + dangerous keyword, treat even low toxicity as harmful
            if has_instruction and has_dangerous:
                if max_prob > 0.0001:  # Accept almost any toxicity for dangerous instructions
                    print(f"Semantic: HARMFUL INSTRUCTION detected")
                    print(f"   Pattern: instruction + dangerous keyword, toxicity: {max_label}={max_prob:.3f}")
                    return "harmful"
            
            # Normal toxicity check
            if max_prob > self.config.semantic_threshold:
                if max_label in ['threat', 'severe_toxic']:
                    print(f"Semantic: HARMFUL content ({max_label}={max_prob:.3f})")
                    return "harmful"
                elif max_label in ['toxic', 'insult', 'obscene', 'identity_hate']:
                    print(f"Semantic: SOCIAL ENGINEERING ({max_label}={max_prob:.3f})")
                    return "social_engineering"
                else:
                    print(f"Semantic: HARMFUL content ({max_label}={max_prob:.3f})")
                    return "harmful"
            
            # Below threshold
            if max_prob > 0.3:
                print(f"Semantic: Borderline ({max_label}={max_prob:.3f})")
            else:
                print(f"Semantic: Safe ({max_label}={max_prob:.3f})")
            
            return None
            
        except Exception as e:
            print(f"Semantic inference failed: {e}")
            return None
    
    def compute_safety_index(self, 
                            dialogue_embedding: torch.Tensor) -> float:
        """
        Compute safety index using NBF models.
        
        Args:
            dialogue_embedding: Embedding of current dialogue state
            
        Returns:
            Safety index (negative = safe, positive = unsafe)
        """
        if self.nbf_models is None:
            # Simulation mode - return mock value
            return np.random.randn() * 0.1
        
        try:
            with torch.no_grad():
                # Extract models
                ssm = self.nbf_models.get('ssm')
                nbf = self.nbf_models.get('nbf')
                
                if ssm is None or nbf is None:
                    return 0.0
                
                # Compute latent state using state-space model
                latent_state = ssm(dialogue_embedding)
                
                # Compute barrier function value
                safety_index = nbf(latent_state)
                
                return float(safety_index.item())
        except Exception as e:
            print(f"Error computing safety index: {e}")
            return 0.0
    
    def modulate_eta(self, 
                    context_type: str,
                    escalation_detected: bool,
                    recent_safety_trajectory: List[float]) -> float:
        """
        Dynamically modulate threshold η based on context.
        
        This is the core innovation addressing the paper's limitation.
        
        Args:
            context_type: Type of conversation context
            escalation_detected: Whether escalation is detected
            recent_safety_trajectory: Recent safety scores
            
        Returns:
            Modulated threshold value
        """
        eta = self.config.base_eta
        
        # Context-based modulation
        if context_type == "creative":
            eta *= self.config.creative_reduction
        elif context_type == "technical":
            eta *= self.config.technical_reduction
        elif context_type in ["medical", "chemical", "social_engineering"]:
            eta *= self.config.high_stakes_increase
        
        # Escalation-based modulation
        if escalation_detected:
            eta *= self.config.escalation_increase
        
        # Trajectory-based modulation
        if len(recent_safety_trajectory) >= 2:
            # Check if safety is degrading
            trend = np.polyfit(
                range(len(recent_safety_trajectory)),
                recent_safety_trajectory,
                1
            )[0]
            
            if trend > 0.01:  # Increasing risk
                eta *= 1.5
        
        # Clamp to bounds
        eta = np.clip(eta, self.config.min_eta, self.config.max_eta)
        
        return eta
    
    def classify_context(self, user_message: str) -> str:
        """
        Classify conversation context for η modulation.
        
        Uses semantic classifier if available, falls back to keywords.
        
        Args:
            user_message: Current user message
            
        Returns:
            Context type string with confidence-based priority
        """
        # Try semantic classifier first
        if self.semantic_classifier is not None:
            try:
                context = self._classify_with_semantic_model(user_message)
                if context is not None:
                    print(f"Semantic: {context}")
                    return context
            except Exception as e:
                if not self.config.fallback_to_keywords:
                    return "general"
        
        # Keyword fallback
        message_lower = user_message.lower()
        
        # Enhanced keyword sets with semantic groupings
        CONTEXT_PATTERNS = {
            "medical": {
                "substances": ["drug", "chemical", "compound", "substance", "medication", 
                             "narcotic", "opioid", "stimulant", "sedative", "hallucinogen",
                             "methamphetamine", "cocaine", "heroin", "fentanyl", "mdma"],
                "medical": ["prescription", "diagnosis", "symptom", "treatment", "therapy",
                           "surgery", "medical procedure", "healthcare", "clinical"],
                "weapons": ["weapon", "explosive", "bomb", "ammunition", "firearm", "gun",
                           "rifle", "pistol", "grenade", "detonator", "ied"],
                "chemistry": ["synthesis", "synthesize", "chemical reaction", "reagent",
                            "catalyst", "precursor", "laboratory", "distillation"]
            },
            "social_engineering": {
                "manipulation": ["manipulate", "deceive", "trick", "fool", "con", "dupe",
                               "mislead", "gaslight", "coerce", "exploit"],
                "attacks": ["phishing", "social engineer", "pretexting", "baiting",
                          "tailgating", "quid pro quo", "impersonate"],
                "fraud": ["scam", "fraud", "steal", "hack", "breach", "unauthorized access",
                         "identity theft", "credential", "password", "bypass security"],
                "harmful_content": ["spread misinformation", "fake news", "propaganda",
                                   "harass", "bully", "threaten", "doxx", "swat"]
            },
            "creative": {
                "writing": ["poem", "poetry", "story", "fiction", "novel", "narrative",
                          "creative writing", "short story", "screenplay", "dialogue"],
                "imagination": ["imagine", "pretend", "fantasy", "hypothetical", "suppose",
                              "what if", "fictional", "make believe", "alternate reality"],
                "art": ["write about", "create a story", "compose", "draft", "author",
                       "literary", "prose", "verse", "character development"]
            },
            "technical": {
                "programming": ["code", "program", "function", "class", "method", "variable",
                              "algorithm", "data structure", "api", "library"],
                "development": ["debug", "implement", "develop", "build", "create application",
                              "software", "compile", "deploy", "refactor", "optimize"],
                "technical": ["technical", "engineering", "architecture", "design pattern",
                            "database", "sql", "query", "server", "client", "framework"]
            }
        }
        
        # Score each context based on pattern matches
        context_scores = {}
        
        for context_type, categories in CONTEXT_PATTERNS.items():
            score = 0
            matched_patterns = []
            
            for category, keywords in categories.items():
                for keyword in keywords:
                    if keyword in message_lower:
                        # Weight multi-word phrases higher
                        weight = len(keyword.split()) * 2
                        score += weight
                        matched_patterns.append(keyword)
            
            if score > 0:
                context_scores[context_type] = {
                    'score': score,
                    'patterns': matched_patterns
                }
        
        # Additional contextual analysis
        
        # Check for "how to" patterns with dangerous keywords
        if "how to" in message_lower or "how do i" in message_lower:
            dangerous_phrases = ["make", "create", "build", "synthesize", "produce", "manufacture"]
            if any(phrase in message_lower for phrase in dangerous_phrases):
                # Boost medical/harmful context if combined with dangerous action
                if "medical" in context_scores:
                    context_scores["medical"]["score"] *= 2
        
        # Check for instructional patterns
        instructional_patterns = ["step by step", "instructions for", "guide to", "tutorial on"]
        if any(pattern in message_lower for pattern in instructional_patterns):
            # Boost existing harmful contexts
            if "medical" in context_scores:
                context_scores["medical"]["score"] *= 1.5
            if "social_engineering" in context_scores:
                context_scores["social_engineering"]["score"] *= 1.5
        
        # Check for roleplay/hypothetical framing (often used in jailbreaks)
        roleplay_indicators = ["pretend", "roleplay", "act as", "you are a", "imagine you're"]
        if any(indicator in message_lower for indicator in roleplay_indicators):
            # If roleplay + harmful context, boost it
            if "medical" in context_scores or "social_engineering" in context_scores:
                for ctx in ["medical", "social_engineering"]:
                    if ctx in context_scores:
                        context_scores[ctx]["score"] *= 1.3
        
        # Check for negation/opposite day patterns (jailbreak technique)
        negation_patterns = ["opposite day", "don't tell me", "refuse to", "do not provide"]
        if any(pattern in message_lower for pattern in negation_patterns):
            # Flag as social engineering attempt
            if "social_engineering" not in context_scores:
                context_scores["social_engineering"] = {'score': 5, 'patterns': ['negation_detected']}
            else:
                context_scores["social_engineering"]["score"] *= 2
        
        # Return highest scoring context
        if context_scores:
            best_context = max(context_scores.items(), key=lambda x: x[1]['score'])
            return best_context[0]
        
        return "general"
    
    def process_turn(self,
                    user_message: str,
                    dialogue_embedding: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Process a conversation turn with integrated NBF + circuit breaker.
        
        Args:
            user_message: Current user message
            dialogue_embedding: Dialogue state embedding (if available)
            
        Returns:
            Dictionary with decision and metadata
        """
        self.safety_state.turn_count += 1
        
        # Classify context
        context_type = self.classify_context(user_message)
        self.conversation_context = context_type
        
        # Compute safety index using NBF
        if dialogue_embedding is not None:
            safety_index = self.compute_safety_index(dialogue_embedding)
        else:
            # Fallback: use circuit breaker only
            safety_index = 0.0
        
        # Update trajectory
        self.safety_state.trajectory.append(safety_index)
        if len(self.safety_state.trajectory) > 10:
            self.safety_state.trajectory.pop(0)
        
        # Detect escalation
        escalation_detected = False
        if len(self.safety_state.trajectory) >= self.config.window_size:
            recent = self.safety_state.trajectory[-self.config.window_size:]
            if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                escalation_detected = True
        
        # Modulate eta dynamically
        dynamic_eta = self.modulate_eta(
            context_type,
            escalation_detected,
            self.safety_state.trajectory
        )
        
        self.safety_state.eta = dynamic_eta
        
        # Apply NBF steering decision
        nbf_unsafe = safety_index > -dynamic_eta
        
        # Integrate with circuit breaker
        circuit_breaker_blocked = False
        circuit_breaker_result = None
        
        if self.circuit_breaker:
            try:
                # Use a dummy function that signals success/failure based on safety
                def check_safety():
                    if nbf_unsafe:
                        raise Exception("NBF detected unsafe content")
                    return True
                
                # Call through circuit breaker
                self.circuit_breaker.call(check_safety)
                circuit_breaker_blocked = False
                
                circuit_breaker_result = {
                    'allowed': True,
                    'circuit_state': self.circuit_breaker.get_state().value
                }
            except CircuitBreakerOpenError:
                circuit_breaker_blocked = True
                self.safety_state.circuit_breaker_triggered = True
                circuit_breaker_result = {
                    'allowed': False,
                    'circuit_state': 'open'
                }
            except Exception:
                # NBF detected unsafe, let circuit breaker record failure
                circuit_breaker_result = {
                    'allowed': False,
                    'circuit_state': self.circuit_breaker.get_state().value
                }
        
        # Combined decision: block if either system flags
        should_block = nbf_unsafe or circuit_breaker_blocked
        self.safety_state.steering_active = should_block
        
        # Prepare result
        result = {
            'allowed': not should_block,
            'safety_index': safety_index,
            'dynamic_eta': dynamic_eta,
            'context_type': context_type,
            'escalation_detected': escalation_detected,
            'nbf_triggered': nbf_unsafe,
            'circuit_breaker_triggered': circuit_breaker_blocked,
            'turn_count': self.safety_state.turn_count,
            'safety_trajectory': self.safety_state.trajectory.copy(),
            'circuit_breaker_result': circuit_breaker_result,
            'reasoning': []
        }
        
        # Add reasoning
        if nbf_unsafe:
            result['reasoning'].append(
                f"NBF safety index {safety_index:.4f} exceeds threshold {-dynamic_eta:.4f}"
            )
        if circuit_breaker_blocked:
            result['reasoning'].append("Circuit breaker triggered")
        if escalation_detected:
            result['reasoning'].append("Escalation pattern detected in conversation")
        if context_type in ["medical", "social_engineering"]:
            result['reasoning'].append(f"High-stakes context detected: {context_type}")
        
        return result
    
    def reset(self):
        """Reset the integration state"""
        self.safety_state = SafetyState(
            safety_index=0.0,
            trajectory=[],
            steering_active=False,
            circuit_breaker_triggered=False,
            eta=self.config.base_eta,
            turn_count=0
        )
        
        if self.circuit_breaker:
            self.circuit_breaker.reset()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            'total_turns': int(self.safety_state.turn_count),
            'steering_active': bool(self.safety_state.steering_active),
            'circuit_breaker_triggered': bool(self.safety_state.circuit_breaker_triggered),
            'current_eta': float(self.safety_state.eta),
            'base_eta': float(self.config.base_eta),
            'eta_modulation_range': (float(self.config.min_eta), float(self.config.max_eta)),
            'safety_trajectory': [float(x) for x in self.safety_state.trajectory],
            'avg_safety_index': float(
                np.mean(self.safety_state.trajectory) 
                if self.safety_state.trajectory else 0.0
            ),
            'conversation_context': str(self.conversation_context)
        }
        
        if self.circuit_breaker:
            cb_stats = self.circuit_breaker.get_stats()
            # Convert CircuitBreakerStats to JSON-serializable dict
            stats['circuit_breaker_stats'] = {
                'total_requests': int(cb_stats.total_requests),
                'total_failures': int(cb_stats.total_failures),
                'total_successes': int(cb_stats.total_successes),
                'state_changes': [(float(timestamp), str(state.value)) for timestamp, state in cb_stats.state_changes],
                'recent_requests': [bool(x) for x in cb_stats.recent_requests]
            }
        
        return stats
