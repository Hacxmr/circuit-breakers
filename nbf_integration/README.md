# NBF-LLM + Circuit Breaker Integration

This module integrates the Neural Barrier Function (NBF) approach from ["Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks"](https://arxiv.org/abs/2503.00187) (Hu, Robey, & Liu) with our context-aware circuit breaker system.

## Overview

The integration addresses key limitations identified in the NBF-LLM paper:

### Limitations Addressed

1. **Fixed Threshold Problem**: The paper notes that "a fixed steering threshold η leads to over-refusal"
   - **Solution**: Dynamic η modulation based on conversation context and threat trajectory

2. **Architectural Generalization**: Limited to specific model architectures
   - **Solution**: Universal safety layer that works across different LLM backends

3. **Online Adaptation**: Static models don't learn from new attacks
   - **Solution**: Circuit breaker provides adaptive defense that learns from patterns

## Architecture

```
User Input
    ↓
┌─────────────────────────────────────┐
│  Dynamic NBF Integration            │
│  ┌────────────────────────────┐    │
│  │ 1. Context Classification  │    │
│  │    - Creative vs Technical │    │
│  │    - High-stakes detection │    │
│  └────────────────────────────┘    │
│                ↓                    │
│  ┌────────────────────────────┐    │
│  │ 2. NBF Safety Index        │    │
│  │    - State-space model     │    │
│  │    - Barrier function      │    │
│  └────────────────────────────┘    │
│                ↓                    │
│  ┌────────────────────────────┐    │
│  │ 3. Dynamic η Modulation    │    │
│  │    - Context-based         │    │
│  │    - Escalation-aware      │    │
│  │    - Trajectory-based      │    │
│  └────────────────────────────┘    │
│                ↓                    │
│  ┌────────────────────────────┐    │
│  │ 4. Circuit Breaker         │    │
│  │    - Multi-turn tracking   │    │
│  │    - Crescendo detection   │    │
│  │    - Adaptive thresholds   │    │
│  └────────────────────────────┘    │
└─────────────────────────────────────┘
    ↓
Allow / Block + Steering
```

## Installation

1. Ensure the NBF-LLM repository is cloned:
```bash
# Already done if you're reading this
cd /Users/hacxmr/Documents/GitHub/circuit-breakers
ls nbf_llm/  # Should see the cloned repository
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r nbf_llm/requirements.txt
```

3. Set up API keys in `.env` file:
```bash
# Create .env in the nbf_llm directory
cd nbf_llm
cat > .env << EOF
BASE_URL_GPT="https://api.openai.com/v1"
GPT_API_KEY="your-openai-key"

BASE_URL_LLAMA="https://api.llama-api.com"
LLAMA_API_KEY="your-llama-key"

CLAUDE_API_KEY="your-claude-key"
EOF
```

## Usage

### Basic NBF with Circuit Breaker

```python
from nbf_integration import DynamicNBFIntegration, NBFConfig

# Configure integration
config = NBFConfig(
    model_path="./nbf_llm/models/models_best_nbf_released.pth",
    base_eta=0.001,
    min_eta=0.0001,
    max_eta=0.01
)

# Initialize
integration = DynamicNBFIntegration(
    nbf_config=config,
    use_circuit_breaker=True
)

# Process conversation turns
for user_message in conversation:
    result = integration.process_turn(user_message)
    
    if result['allowed']:
        # Safe to send to LLM
        response = llm.generate(user_message)
    else:
        # Blocked - log reasoning
        print(f"Blocked: {result['reasoning']}")
        print(f"Safety index: {result['safety_index']:.4f}")
        print(f"Dynamic η: {result['dynamic_eta']:.6f}")
```

### Run Integrated Steering Against Attacks

```bash
cd nbf_integration

# Test against crescendo attacks with circuit breaker
python integrated_steering.py \
    --attack_method crescendomation \
    --target_model gpt-4o \
    --use_circuit_breaker \
    --dynamic_eta \
    --safety_filtering \
    --threshold 0.001

# Test against opposite_day attacks
python integrated_steering.py \
    --attack_method opposite_day \
    --target_model gpt-4o \
    --use_circuit_breaker \
    --dynamic_eta \
    --min_eta 0.0001 \
    --max_eta 0.01

# Test against actor_attack
python integrated_steering.py \
    --attack_method actor_attack \
    --target_model claude-3-5-sonnet-20241022 \
    --use_circuit_breaker \
    --dynamic_eta
```

## Dynamic η Modulation

The key innovation is context-aware threshold adaptation:

| Context Type | η Modifier | Rationale |
|-------------|-----------|-----------|
| Creative (poems, stories) | ×0.5 | Reduce over-refusal for creative tasks |
| Technical (coding) | ×0.3 | Allow legitimate programming help |
| Medical/Chemical | ×3.0 | High stakes require strict filtering |
| Social Engineering | ×3.0 | Manipulation attempts need vigilance |
| Escalation Detected | ×2.0 | Crescendo pattern identified |

## Evaluation Metrics

The system tracks:

1. **Safety Invariance**: Maintains safety across 10+ turns
2. **Over-Refusal Rate (ORR)**: Reduced through dynamic η
3. **Attack Success Rate (ASR)**: Should remain low
4. **Circuit Breaker Triggers**: Multi-turn pattern detection
5. **η Trajectory**: How threshold adapts over conversation

## Key Features

### 1. Dynamic Threshold Modulation

Instead of fixed η = 0.001, the system adapts:
- Creative context: η = 0.0005 (more permissive)
- Medical context: η = 0.003 (more restrictive)
- Escalation detected: η doubles

### 2. Multi-Layer Defense

- Layer 1: NBF safety index computation
- Layer 2: Dynamic η modulation
- Layer 3: Circuit breaker pattern detection
- Layer 4: Integrated decision with reasoning

### 3. Crescendo Attack Detection

Specifically designed to counter multi-turn escalation:
- Tracks safety trajectory over conversation
- Detects gradual degradation patterns
- Increases vigilance adaptively

### 4. Cross-Architecture Support

Works with any LLM backend:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Meta (Llama)
- Local models (via API)

## Research Extensions

### Implement Online Learning

```python
# Pseudo-code for adversarial training loop
attacker = AdversarialAgent()
defender = DynamicNBFIntegration()

for episode in range(num_episodes):
    # Attacker tries to find vulnerabilities
    attack_trajectory = attacker.generate_attack()
    
    # Defender processes and logs near-misses
    for turn in attack_trajectory:
        result = defender.process_turn(turn)
        if result['safety_index'] > -0.002:  # Near-miss
            defender.update_from_near_miss(turn)
    
    # Update attacker strategy
    attacker.update_from_defenses(defender.get_patterns())
```

### Cross-Architecture Transfer

```python
# Train on one architecture, deploy on another
nbf_trained_on_llama = load_nbf("llama3-8b")

# Transfer to GPT-4
universal_embeddings = extract_universal_embeddings(gpt4_states)
safety_index = nbf_trained_on_llama.predict(universal_embeddings)
```

## Comparison with Original NBF

| Feature | Original NBF | NBF + Circuit Breaker |
|---------|-------------|----------------------|
| Threshold | Fixed η | Dynamic η (context-aware) |
| Multi-turn | Pattern detection | Pattern + circuit breaker |
| Adaptation | Static | Online learning capable |
| Over-refusal | Higher | Reduced via context |
| Architecture | Model-specific | Universal layer |

## Results Structure

Output files contain:
```json
{
  "Goal": "original harmful request",
  "case_stats": {
    "total_turns": 5,
    "current_eta": 0.002,
    "circuit_breaker_triggered": true,
    "steering_active": true,
    "safety_trajectory": [-0.1, 0.0, 0.15, 0.3, 0.45],
    "conversation_context": "social_engineering"
  }
}
```

## Future Work

1. **Synonymic Reformulation Defense**: Enhance semantic understanding to catch paraphrased attacks
2. **Universal Latent Mapping**: Create architecture-agnostic safety representations
3. **Real-time Adaptation**: Implement continuous learning from detected attacks
4. **Attention Shifting Detection**: Track focus changes in multi-turn conversations

## References

- [NBF-LLM Paper](https://arxiv.org/abs/2503.00187)
- [NBF-LLM GitHub](https://github.com/HanjiangHu/NBF-LLM)
- [ActorAttack](https://github.com/AI45Lab/ActorAttack)
- [Automated Multi-Turn Jailbreaks](https://github.com/AIM-Intelligence/Automated-Multi-Turn-Jailbreaks)

## Citation


```bibtex
@article{hu2025steering,
  title={Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks},
  author={Hu, Hanjiang and Robey, Alexander and Liu, Hamed Hassani},
  journal={arXiv preprint arXiv:2503.00187},
  year={2025}
}
```
