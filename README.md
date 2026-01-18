# Circuit Breaker for Jailbreak Prevention

This project implements a circuit breaker pattern specifically designed for preventing jailbreak attacks in Large Language Models (LLMs). The circuit breaker monitors interactions and detects patterns indicative of jailbreak attempts, temporarily halting or modifying responses when suspicious behavior is detected.

Note: Local execution with LM Studio or Ollama is now supported. See [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for details.

## Overview

The circuit breaker pattern operates in three states:
- **Closed**: Normal operation, monitoring for suspicious patterns
- **Open**: Circuit is open, blocking potentially harmful requests after threshold exceeded
- **Half-Open**: Testing state, allowing limited requests to check if the system has recovered

## Literature References

This implementation is inspired by and follows research from:

1. **Active Honeypot Guardrail System** (arXiv:2510.15017)
   - Proactive defense using bait responses to probe user intent
   - Disrupts multi-turn jailbreak attempts

2. **EEG-Defender** (arXiv:2408.11308)
   - Early exit generation to detect malicious inputs
   - Terminates generation immediately upon detection

3. **AdaSteer** (arXiv:2504.09466)
   - Adaptive activation steering method
   - Dynamically adjusts model behavior based on input characteristics

4. **Concept Enhancement Engineering (CEE)** (arXiv:2504.13201)
   - Dynamic steering of internal activations
   - Reinforces safe behavior during inference

## Project Structure

```
circuit-breaker/
├── README.md
├── requirements.txt
├── circuit_breaker/
│   ├── __init__.py
│   ├── breaker.py          # Core circuit breaker implementation
│   ├── detector.py         # Jailbreak detection mechanisms
│   └── metrics.py          # Evaluation metrics
├── experiments/
│   ├── __init__.py
│   ├── test_cases.py       # Test cases for jailbreak attempts
│   └── evaluator.py        # Experiment evaluation framework
└── examples/
    └── basic_usage.py      # Example usage
```

## Installation

### Local Execution with LM Studio or Ollama

```bash
# LM Studio: Start server in LM Studio UI (localhost:1234)
# Ollama: Install and start service
brew install ollama  # macOS
ollama serve
ollama pull llama3.2

# Run experiments
cd nbf_integration
python3 integrated_steering.py \
    --attack_method crescendomation \
    --target_model llama3.2 \
    --use_circuit_breaker \
    --dynamic_eta
```

See [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for detailed instructions.

### Cloud-Based Execution

```bash
pip install -r requirements.txt
pip install -r nbf_llm/requirements.txt

# Configure API keys in nbf_llm/.env
cd nbf_llm
cp .env.example .env  # Edit with your API keys
```

## Usage

See `examples/basic_usage.py` for detailed examples.

## Experiments

Run experiments to evaluate the effectiveness of the circuit breaker:

```bash
python -m experiments.evaluator
```

This will test the circuit breaker on various jailbreak patterns and report:
- Precision, Recall, F1 Score
- False Positive/Negative Rates
- Circuit breaker state transitions
- Detection statistics

## Tuning Detection

The detection system can be tuned by adjusting:

1. **Scoring Weights** (in `circuit_breaker/detector.py`):
   - Pattern matching: 40% weight
   - Keyword detection: 30% weight
   - Bypass techniques: 20% weight
   - Multi-turn analysis: 10% weight

2. **Detection Thresholds**:
   - `is_jailbreak` threshold (currently 0.3 or MEDIUM threat level)
   - Threat level boundaries (SAFE, LOW, MEDIUM, HIGH, CRITICAL)

3. **Circuit Breaker Configuration**:
   - `failure_threshold`: Number of failures before opening
   - `failure_rate_threshold`: Failure rate to trigger opening
   - `timeout`: Time before attempting recovery

See `QUICKSTART.md` for detailed configuration options.

## License

MIT

