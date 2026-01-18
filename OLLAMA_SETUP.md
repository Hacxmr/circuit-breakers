# Running Circuit Breaker with Local Models

This guide shows you how to run the NBF circuit breaker integration with local models using LM Studio or Ollama.

## Option 1: LM Studio

LM Studio provides a user-friendly interface for running local models.

### Installation

Download from: https://lmstudio.ai/

### Setup

1. Launch LM Studio
2. Go to "Discover" tab and download models (e.g., Llama 3.2, Mistral, Qwen)
3. Go to "Local Server" tab
4. Select a model and click "Start Server"
5. Server runs on http://localhost:1234 by default

### Run Experiments

First, check what models you have:
```bash
curl http://localhost:1234/v1/models
```

Then run with the exact model ID:
```bash
cd nbf_integration

# Example: phi-3.5-mini-instruct
python3 integrated_steering.py \
    --attack_method crescendomation \
    --target_model "phi-3.5-mini-instruct" \
    --use_circuit_breaker \
    --dynamic_eta

# Example: llama-3.2-3b-instruct  
python3 integrated_steering.py \
    --attack_method crescendomation \
    --target_model "llama-3.2-3b-instruct" \
    --use_circuit_breaker \
    --dynamic_eta
```

Note: First run takes 30-60 seconds to load dependencies. Use model names from the curl command above.

## Option 2: Ollama

Ollama provides command-line access to local models.

### Install Ollama

macOS:
```bash
brew install ollama
```

Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Start Ollama Service

```bash
ollama serve
```

This starts the Ollama API server on http://localhost:11434

### Pull Models

```bash
ollama pull llama3.2          # 3B model
ollama pull mistral           # 7B model
ollama pull qwen2.5:7b        # 7B model
ollama pull gemma2:9b         # 9B model
ollama pull llama3.1:8b       # 8B model
```

List available models:
```bash
ollama list
```

## Configuration

No .env file is needed by default. The system automatically detects:
- LM Studio at http://localhost:1234/v1
- Ollama at http://localhost:11434/v1

To customize, create nbf_llm/.env:
```bash
cd nbf_llm
cat > .env << EOF
LMSTUDIO_BASE_URL=http://localhost:1234/v1
OLLAMA_BASE_URL=http://localhost:11434/v1
EOF
```

## Running Experiments

### With LM Studio

```bash
cd nbf_integration

# Use the exact model identifier from LM Studio
python integrated_steering.py \
    --attack_method crescendomation \
    --target_model "llama-3.2-3b-instruct" \
    --use_circuit_breaker \
    --dynamic_eta
```

### With Ollama

```bash
cd nbf_integration

# Using llama3.2
python integrated_steering.py \
    --attack_method crescendomation \
    --target_model llama3.2 \
    --use_circuit_breaker \
    --dynamic_eta \
    --safety_filtering

# Using mistral
python integrated_steering.py \
    --attack_method crescendomation \
    --target_model mistral \
    --use_circuit_breaker \
    --dynamic_eta

# Using qwen2.5
python integrated_steering.py \
    --attack_method crescendomation \
    --target_model qwen2.5:7b \
    --use_circuit_breaker \
    --dynamic_eta
```

### Test Different Attacks

Opposite Day Attack:
```bash
python integrated_steering.py \
    --attack_method opposite_day \
    --target_model llama3.2 \
    --use_circuit_breaker \
    --dynamic_eta
```

Actor Attack:
```bash
python integrated_steering.py \
    --attack_method actor_attack \
    --target_model mistral \
    --use_circuit_breaker \
    --dynamic_eta
```

Acronym Attack:
```bash
python integrated_steering.py \
    --attack_method acronym \
    --target_model qwen2.5:7b \
    --use_circuit_breaker \
    --dynamic_eta
```

### Adjust Circuit Breaker Parameters

```bash
python integrated_steering.py \
    --attack_method crescendomation \
    --target_model llama3.2 \
    --use_circuit_breaker \
    --dynamic_eta \
    --min_eta 0.0001 \
    --max_eta 0.01 \
    --threshold 0.001 \
    --safety_filtering
```

## Model Selection Guide

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| llama3.2 | 3B | Very Fast | Good | Quick testing, rapid iteration |
| mistral | 7B | Fast | Very Good | Balanced testing |
| qwen2.5:7b | 7B | Fast | Excellent | Production-like testing |
| gemma2:9b | 9B | Medium | Excellent | Safety-focused testing |
| llama3.1:8b | 8B | Medium | Excellent | General purpose |

## Troubleshooting

### Error: "Connection refused"

Make sure Ollama is running:
```bash
ollama serve
```

### Error: "Model not found"

Pull the model first:
```bash
ollama pull llama3.2
```

### Slow Performance

- Use smaller models (llama3.2, mistral)
- Reduce context window if needed
- Close other applications

### Check Ollama Status

```bash
# See running models
ollama ps

# Check if Ollama is responding
curl http://localhost:11434/api/tags
```

## Performance Tips

1. **First run is slower**: Ollama loads models into memory on first use
2. **Keep Ollama running**: Models stay in memory between requests
3. **Monitor resources**: 
   ```bash
   # macOS
   top -pid $(pgrep ollama)
   
   # Linux
   htop -p $(pgrep ollama)
   ```

## Example Output

```bash
$ python integrated_steering.py --attack_method crescendomation --target_model llama3.2 --use_circuit_breaker

Ollama client initialized at http://localhost:11434/v1
Circuit breaker integration enabled
Dynamic η modulation enabled

Testing case 1/50...
  Turn 1: ALLOWED (safety_index: -0.05, η: 0.0010)
  Turn 2: ALLOWED (safety_index: -0.03, η: 0.0010)
  Turn 3: BLOCKED (safety_index: 0.15, η: 0.0020) - Circuit breaker triggered
  
Results saved to: results/crescendomation_llama3.2_20260113_143022.json
```

## Comparing Local vs Cloud Models

Run the same test with both:

```bash
# Local with Ollama
python integrated_steering.py \
    --attack_method crescendomation \
    --target_model llama3.2 \
    --use_circuit_breaker

# Cloud with GPT-4 (requires API key in .env)
python integrated_steering.py \
    --attack_method crescendomation \
    --target_model gpt-4o \
    --use_circuit_breaker
```

## Benefits of Local Execution

- Privacy: All data stays on your machine
- Cost: No API fees
- Speed: No network latency
- Availability: Works offline
- Control: Experiment freely without quotas

## Next Steps

1. Experiment with different models: Try various model sizes to find the best balance
2. Tune circuit breaker: Adjust `--min_eta` and `--max_eta` for your use case
3. Create custom attacks: Test your own attack scenarios
4. Benchmark: Compare defense effectiveness across models

## Advanced: Using Ollama with Original NBF Steering

You can also use Ollama with the original NBF steering script:

```bash
cd nbf_llm

python steering.py \
    --attack_method crescendomation \
    --target_model llama3.2 \
    --safety_filtering \
    --threshold 0.001
```

The Ollama integration works with both scripts!
