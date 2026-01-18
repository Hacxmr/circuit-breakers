# Circuit Breaker + NBF-LLM Integration - Changelog & Documentation

## Overview

This document tracks all changes made to integrate the NBF-LLM framework with LM Studio/Ollama for local model execution and circuit breaker functionality.

## Version History

### v1.1.0 - 2026-01-14
**Fixed: Statistics Tracking in Output Files**

**Problem:** The `case_stats` section in output JSONL files showed incorrect values:
- `total_turns: 0` (should show actual turn count like 2, 3, etc.)
- `steering_active: false` (even when steering occurred)
- `circuit_breaker_triggered: false` (even when triggered)
- `safety_trajectory: []` (empty instead of containing safety values)

**Root Cause:** The `integration` object (which tracks these stats) was never passed to the attack run functions, so `integration.process_turn()` was never called.

**Solution:** Modified all attack functions to accept and use the `integration` parameter:

**Files Modified:**
- `nbf_integration/integrated_steering.py` - Pass integration to all attack functions
- `nbf_llm/attacks/opposite_day/run.py` - Accept integration, call `integration.process_turn(prompt)`
- `nbf_llm/attacks/crescendomation/run.py` - Accept integration, call `integration.process_turn(prompt)`
- `nbf_llm/attacks/actor_attack/run.py` - Accept integration in `run_actor_attack()`, `attack_single()`, `call_multi()`, and `summary()`
- `nbf_llm/attacks/acronym/run.py` - Accept integration, call `integration.process_turn(prompt)`
- `nbf_integration/__init__.py` - Convert numpy types to native Python types in `get_statistics()` for JSON serialization

**Testing:** After this fix, `case_stats` now correctly shows:
```json
{
  "total_turns": 2,
  "steering_active": true,
  "circuit_breaker_triggered": false,
  "current_eta": 0.003,
  "base_eta": 0.001,
  "eta_modulation_range": [0.0001, 0.01],
  "safety_trajectory": [0.0, 0.0],
  "avg_safety_index": 0.0,
  "conversation_context": "medical"
}
```

**Note:** In simulation mode (when NBF model file is missing), `safety_trajectory` contains simulated random values. With actual NBF models loaded, these would contain real safety index calculations.

---

### v1.0.0 - 2026-01-14
**Initial Local Model Support & Bug Fixes**

## Changes Made

### 1. Local Model Support (LM Studio & Ollama)

**Files Modified:**
- `nbf_integration/integrated_steering.py`
- `nbf_llm/attacks/utils/generate.py`
- `.env.example`

**Changes:**
- Added LM Studio client initialization (http://localhost:1234/v1)
- Added Ollama client initialization (http://localhost:11434/v1)
- Updated `get_client()` to prioritize local models over cloud APIs
- Added JSON parsing fallback for local models that don't support JSON mode properly

**Code:**
```python
# LM Studio client initialization
lmstudio_base_url = get_env_variable('LMSTUDIO_BASE_URL') or 'http://localhost:1234/v1'
clients['lmstudio'] = OpenAI(base_url=lmstudio_base_url, api_key='lmstudio')

# Ollama client initialization  
ollama_base_url = get_env_variable('OLLAMA_BASE_URL') or 'http://localhost:11434/v1'
clients['ollama'] = OpenAI(base_url=ollama_base_url, api_key='ollama')
```

### 2. JSON Parsing Fixes for Local Models

**Files Modified:**
- `nbf_llm/attacks/utils/generate.py`
- `nbf_llm/attacks/crescendomation/run.py`
- `nbf_llm/attacks/opposite_day/run.py`
- `nbf_llm/attacks/acronym/run.py`
- `nbf_llm/attacks/actor_attack/run.py`
- `nbf_llm/attacks/utils/check_refusal.py`
- `nbf_llm/attacks/utils/evaluate_with_rubric.py`

**Problem:** Local models (LM Studio/Ollama) don't properly support `json_format=True` and return strings instead of parsed JSON objects.

**Solution:** Added fallback parsing:
```python
# In generate.py
if json_format:
    try:
        return json.loads(content)
    except:
        # Extract JSON from markdown code blocks if present
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        try:
            return json.loads(content)
        except:
            return content  # Return as string if all parsing fails

# In attack functions
if isinstance(data, str):
    return data[:500], "Unable to parse response"

return data.get('generatedQuestion', data.get('question', '')), \
       data.get('lastResponseSummary', data.get('summary', ''))
```

### 3. Default Test Cases Configuration

**File Modified:**
- `nbf_integration/integrated_steering.py`

**Changes:**
- Changed default `--num_cases` from `None` (all 200 cases) to `5`
- Fixed test case counter display to show actual count: `Test Case 1/5` instead of `Test Case 1/200`

**Before:**
```python
parser.add_argument("--num_cases", type=int, default=None, help="...")
num_to_process = args.num_cases if args.num_cases else len(test_case)
print(f"Test Case {i+1}/{len(test_case)}: ...")
```

**After:**
```python
parser.add_argument("--num_cases", type=int, default=5, help="...")
num_to_process = args.num_cases
end_index = min(args.start_from + num_to_process, len(test_case))
print(f"Test Case {i+1}/{end_index}: ...")
```

### 4. Attacker Model Default Fix

**File Modified:**
- `nbf_integration/integrated_steering.py`

**Problem:** Default attacker_model was `gpt-4o` which requires API key, causing errors with local models.

**Solution:** Changed default to `None` and use target_model as attacker if not specified:
```python
parser.add_argument("--attacker_model", type=str, default=None, 
                   help="Attacker model name (defaults to same as target_model)")

# Later in code
if args.attacker_model is None:
    args.attacker_model = args.target_model
```

### 5. Documentation Updates

**Files Created:**
- `OLLAMA_SETUP.md` - Comprehensive guide for LM Studio and Ollama setup
- `RUN_COMMANDS.txt` - Quick reference with exact commands for user's models
- `.env.example` - Environment variable template

**Files Modified:**
- `README.md` - Added note about local execution support

## Known Issues

### 1. Case Statistics Not Recording Properly

**Issue:** The `case_stats` in output JSONL files show incorrect values:
```json
{"case_stats": {
  "total_turns": 0,  // Should be actual turn count
  "steering_active": false,
  "circuit_breaker_triggered": false,
  "safety_trajectory": [],  // Should contain safety indices
  "avg_safety_index": 0.0
}}
```

**Root Cause:** The attack run functions (`run_opposite_day`, `run_crescendomation`, etc.) don't have access to the `integration` object to properly track statistics.

**Needs Fix:** Pass integration object to attack functions and add stat tracking after completion.

### 2. NBF Model Path

**Issue:** Warning shows: `Warning: NBF model not found at ./nbf_llm/models/models_best_nbf_released.pth`

**Impact:** Operating in simulation mode without actual NBF steering.

**Solution:** Either:
- Download the pre-trained NBF model and place it at specified path
- Or continue in simulation mode for testing circuit breaker logic

## Usage Examples

### Quick Test (5 cases, default)
```bash
cd nbf_integration
python3 integrated_steering.py \
    --attack_method opposite_day \
    --target_model "phi-3.5-mini-instruct" \
    --use_circuit_breaker
```

### Single Case Test
```bash
python3 integrated_steering.py \
    --attack_method opposite_day \
    --target_model "phi-3.5-mini-instruct" \
    --use_circuit_breaker \
    --num_cases 1
```

### With Dynamic Eta Modulation
```bash
python3 integrated_steering.py \
    --attack_method opposite_day \
    --target_model "meta-llama-3.1-8b-instruct" \
    --use_circuit_breaker \
    --dynamic_eta \
    --min_eta 0.0001 \
    --max_eta 0.01
```

### All Available Models
Based on your LM Studio setup:
- phi-3.5-mini-instruct
- meta-llama-3-8b-instruct-abliterated-v3
- meta-llama-3.1-8b-instruct
- llama-3.2-3b-instruct

## Output Files

Results are saved to: `nbf_integration/results/`

Format: `{attack_method}_{model}_{suffix}_{timestamp}.jsonl`

Each line contains:
```json
{"Goal": "harmful request"}
{"round": 1, "user": "...", "assistant": "...", "score": 1}
{"round": 2, "user": "...", "assistant": "...", "score": 5}
{"goal_achieved": true}
{"case_stats": {...}}
```

## Performance Notes

- **First run**: Takes 30-60 seconds to load dependencies (torch, transformers, etc.)
- **Model loading**: Keep model loaded in LM Studio to avoid "Model unloaded" errors
- **Test duration**: Each test case takes 2-5 minutes depending on model speed and turns
- **Default 5 cases**: Approximately 10-25 minutes total

## Environment Setup

### LM Studio
1. Download from https://lmstudio.ai/
2. Open LM Studio
3. Go to "Local Server" tab
4. Load a model
5. Click "Start Server"
6. Server runs on http://localhost:1234

### Ollama
1. Install: `brew install ollama` (macOS)
2. Start: `ollama serve`
3. Pull model: `ollama pull llama3.2`
4. Server runs on http://localhost:11434

### No Configuration Needed
Both LM Studio and Ollama are auto-detected. No API keys required.

## Command Line Arguments

Full list:
```bash
--target_model          # Model to test (required)
--attacker_model        # Model for attacks (defaults to target_model)
--attack_method         # opposite_day, crescendomation, actor_attack, acronym
--num_cases             # Number of test cases (default: 5)
--start_from            # Starting index (default: 0)
--use_circuit_breaker   # Enable circuit breaker
--dynamic_eta           # Enable dynamic threshold modulation
--min_eta               # Minimum threshold (default: 0.0001)
--max_eta               # Maximum threshold (default: 0.01)
--threshold             # Base threshold η (default: 0.001)
--safety_filtering      # Enable NBF-based steering
--add_safety_index      # Save safety scores in results
```

## Future Improvements Needed

1. **Fix case_stats tracking**: Pass integration object to attack functions
2. **Add progress indicators**: Show estimated time remaining
3. **Improve JSON parsing**: Better extraction from markdown and malformed JSON
4. **Add result analysis**: Summary statistics after run completes
5. **Implement actual NBF steering**: Download or train NBF models
6. **Add graceful interruption**: Save progress on Ctrl+C

## Testing Checklist

- [x] LM Studio client initialization
- [x] Ollama client initialization  
- [x] Local model JSON parsing fallback
- [x] Default num_cases set to 5
- [x] Test case counter shows correct total
- [x] Attacker model defaults to target model
- [ ] Case stats properly recorded
- [ ] NBF model loaded and working
- [ ] Dynamic eta modulation functional
- [ ] Circuit breaker triggers recorded

## Version History

**v1.0 (2026-01-14)**
- Initial integration with LM Studio and Ollama
- JSON parsing fixes for local models
- Default configuration improvements
- Comprehensive documentation

**Known Version:**
- Python 3.9
- torch (installed)
- transformers (installed)
- openai (installed)
- anthropic (installed)
