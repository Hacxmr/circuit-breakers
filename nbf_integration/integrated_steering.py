"""
Modified NBF Steering with Circuit Breaker Integration

This script extends the original NBF-LLM steering.py to integrate
with the circuit breaker, providing multi-layered defense against
crescendo and other multi-turn jailbreaking attacks.

Key enhancements:
1. Dynamic η modulation based on conversation context
2. Circuit breaker triggers on sustained threats
3. Cross-system safety coordination
4. Enhanced logging and metrics
"""

import sys
from pathlib import Path
import os
import datetime
import argparse
import json
from dotenv import load_dotenv

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'nbf_llm'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from attacks.utils import generate
from attacks.crescendomation.run import run_crescendomation
from attacks.opposite_day.run import run_opposite_day
from attacks.actor_attack.run import run_actor_attack
from attacks.acronym.run import run_acronym
import anthropic
from openai import OpenAI

# Import integration module
from nbf_integration import DynamicNBFIntegration, NBFConfig


# Load the API keys from the .env file
load_dotenv()
clients = {}


def get_env_variable(var_name):
    """Fetch environment variable or return None if not set."""
    return os.getenv(var_name)


def initialize_clients():
    """Dynamically initialize available clients based on environment variables."""
    try:
        # LM Studio client (local, no API key needed)
        lmstudio_base_url = get_env_variable('LMSTUDIO_BASE_URL') or 'http://localhost:1234/v1'
        try:
            clients['lmstudio'] = OpenAI(base_url=lmstudio_base_url, api_key='lmstudio')
            print(f"LM Studio client initialized at {lmstudio_base_url}")
        except Exception as e:
            print(f"LM Studio initialization skipped: {e}")
        
        # Ollama client (local, no API key needed)
        ollama_base_url = get_env_variable('OLLAMA_BASE_URL') or 'http://localhost:11434/v1'
        try:
            clients['ollama'] = OpenAI(base_url=ollama_base_url, api_key='ollama')
            print(f"Ollama client initialized at {ollama_base_url}")
        except Exception as e:
            print(f"Ollama initialization skipped: {e}")
        
        gpt_api_key = get_env_variable('GPT_API_KEY')
        gpt_base_url = get_env_variable('BASE_URL_GPT')
        if gpt_api_key and gpt_base_url:
            clients['gpt'] = OpenAI(base_url=gpt_base_url, api_key=gpt_api_key)

        claude_api_key = get_env_variable('CLAUDE_API_KEY')
        if claude_api_key:
            clients['claude'] = anthropic.Anthropic(api_key=claude_api_key)
        
        llama_api_key = get_env_variable('LLAMA_API_KEY')
        llama_base_url = get_env_variable('BASE_URL_LLAMA')
        if llama_api_key and llama_base_url:
            clients['llama'] = OpenAI(base_url=llama_base_url, api_key=llama_api_key)

        if not clients:
            print("No valid API credentials or Ollama found. Exiting.")
            exit(1)

    except Exception as e:
        print(f"Error during client initialization: {e}")
        exit(1)


def get_client(model_name):
    """Select appropriate client based on the given model name."""
    if 'gpt' in model_name or 'o1' in model_name or 'o3' in model_name:
        client = clients.get('gpt') 
    elif 'claude' in model_name:
        client = clients.get('claude')
    elif 'llama' in model_name or 'deepseek' in model_name:
        # Check if it's local first (LM Studio, then Ollama)
        if clients.get('lmstudio'):
            return clients.get('lmstudio')
        if clients.get('ollama'):
            return clients.get('ollama')
        client = clients.get('llama')
    else:
        # Default to local clients for unknown models (LM Studio first, then Ollama)
        if clients.get('lmstudio'):
            return clients.get('lmstudio')
        if clients.get('ollama'):
            return clients.get('ollama')
        raise ValueError(f"Unsupported or unknown model name: {model_name}")

    if not client:
        raise ValueError(f"{model_name} client is not available.")
    return client 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NBF-LLM with Circuit Breaker Integration"
    )
    
    # Original NBF arguments
    parser.add_argument(
        "--target_model", 
        type=str, 
        default="gpt-4o",
        help="Target (victim) model name"
    )
    parser.add_argument(
        "--attacker_model", 
        type=str, 
        default=None,
        help="Attacker model name (defaults to same as target_model)"
    )
    parser.add_argument(
        "--attack_method", 
        type=str, 
        default="crescendomation",
        help="Multi-turn jailbreaking attack method"
    )
    parser.add_argument(
        "--safety_filtering", 
        action='store_true',
        help="Enable NBF-based safety steering"
    )
    parser.add_argument(
        "--add_safety_index", 
        action='store_true',
        help="Save safety scores in results"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="nbf_llm/models/models_best_nbf_released.pth",
        help="Path to pre-trained NBF models"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.001,
        help="Base steering threshold (η)"
    )
    parser.add_argument(
        "--start_from", 
        type=int, 
        default=0,
        help="Starting index in test cases"
    )
    parser.add_argument(
        "--num_cases",
        type=int,
        default=5,
        help="Number of test cases to run (default: 5)"
    )
    parser.add_argument(
        "--save_suffix", 
        type=str, 
        default="",
        help="Suffix for output filename"
    )
    
    # New arguments for circuit breaker integration
    parser.add_argument(
        "--use_circuit_breaker",
        action='store_true',
        help="Enable circuit breaker integration"
    )
    parser.add_argument(
        "--dynamic_eta",
        action='store_true',
        help="Enable dynamic η modulation based on context"
    )
    parser.add_argument(
        "--min_eta",
        type=float,
        default=0.0001,
        help="Minimum threshold for dynamic modulation"
    )
    parser.add_argument(
        "--max_eta",
        type=float,
        default=0.01,
        help="Maximum threshold for dynamic modulation"
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=1,
        help="Maximum number of attack rounds (default: 1 to prevent context overflow)"
    )
    
    # Semantic classifier arguments
    parser.add_argument(
        "--use_semantic_classifier",
        action='store_true',
        help="Enable semantic classifier instead of keyword matching"
    )
    parser.add_argument(
        "--semantic_model",
        type=str,
        default="unitary/toxic-bert",
        help="HuggingFace model name for semantic classification"
    )
    parser.add_argument(
        "--semantic_threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for semantic classifier (0.0-1.0)"
    )
    parser.add_argument(
        "--no_keyword_fallback",
        action='store_true',
        help="Disable fallback to keyword matching if semantic model fails"
    )
    
    args = parser.parse_args()

    # Initialize clients
    initialize_clients()

    # Setup target and attacker models
    target_client = get_client(args.target_model)
    target_model = args.target_model
    target_generate = lambda messages, **kwargs: generate(
        messages, client=target_client, model=target_model, **kwargs
    )

    # Use target model as attacker if not specified
    if args.attacker_model is None:
        args.attacker_model = args.target_model
    
    attacker_client = get_client(args.attacker_model) 
    attacker_model = args.attacker_model
    attacker_generate = lambda messages, **kwargs: generate(
        messages, client=attacker_client, model=attacker_model, **kwargs
    )

    # Initialize integrated NBF + Circuit Breaker system
    if args.use_circuit_breaker or args.dynamic_eta:
        nbf_config = NBFConfig(
            model_path=args.model_path,
            base_eta=args.threshold,
            min_eta=args.min_eta,
            max_eta=args.max_eta,
            use_semantic_classifier=args.use_semantic_classifier,
            semantic_model_name=args.semantic_model,
            semantic_threshold=args.semantic_threshold,
            fallback_to_keywords=not args.no_keyword_fallback
        )
        
        integration = DynamicNBFIntegration(
            nbf_config=nbf_config,
            use_circuit_breaker=args.use_circuit_breaker
        )
        
        
        print("NBF-LLM + CIRCUIT BREAKER INTEGRATION")
        
        print(f"Dynamic η modulation: {args.dynamic_eta}")
        print(f"Circuit breaker enabled: {args.use_circuit_breaker}")
        print(f"Base threshold (η): {args.threshold}")
        print(f"Dynamic range: [{args.min_eta}, {args.max_eta}]")
        print(f"Semantic classifier: {args.use_semantic_classifier}")
        if args.use_semantic_classifier:
            print(f"  Model: {args.semantic_model}")
            print(f"  Threshold: {args.semantic_threshold}")
            print(f"  Keyword fallback: {not args.no_keyword_fallback}")
        
    else:
        integration = None
        print("\nRunning standard NBF steering (no circuit breaker integration)\n")

    # Load test cases
    test_file = Path(__file__).parent.parent / 'nbf_llm' / 'data' / 'test' / 'harmbench_tasks.json'
    with open(test_file, 'r') as f:
        test_case = json.load(f)

    # Prepare output file
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    suffix_parts = [args.save_suffix]
    if args.safety_filtering:
        suffix_parts.append(f"steering_eta{args.threshold}")
    if args.use_circuit_breaker:
        suffix_parts.append("circuit_breaker")
    if args.dynamic_eta:
        suffix_parts.append("dynamic_eta")
    
    final_suffix = "_".join(filter(None, suffix_parts))
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    output_file_path = output_dir / f"{args.attack_method}_{args.target_model}_{final_suffix}_{current_time}.jsonl"
    
    print(f"Output file: {output_file_path}\n")

    # Process each test case
    num_to_process = args.num_cases
    end_index = min(args.start_from + num_to_process, len(test_case))
    
    for i, each_case in enumerate(test_case):
        if i < args.start_from:
            continue
        if i >= end_index:
            break
        
        # Override max_rounds with command line argument for faster testing
        each_case = each_case.copy()  # Don't modify original
        each_case['max_rounds'] = args.max_rounds
        
        print(f"Test Case {i+1}/{end_index}: {each_case['task'][:60]}...")
        print(f"Max rounds: {each_case['max_rounds']}")
        
        
        # Write goal to output
        with open(output_file_path, 'a') as f:
            f.write(json.dumps({"Goal": each_case["task"]}) + '\n')
        
        # Reset integration state for each case
        if integration:
            integration.reset()
        
        # Run attack with integration
        if args.attack_method == "opposite_day":
            run_opposite_day(
                each_case, output_file_path, target_generate, 
                attacker_generate, args, integration
            )
        elif args.attack_method == "crescendomation":
            run_crescendomation(
                each_case, output_file_path, target_generate, 
                attacker_generate, args, integration
            )
        elif args.attack_method == "actor_attack":
            run_actor_attack(
                each_case, output_file_path, target_generate, 
                attacker_generate, args, integration
            )
        elif args.attack_method == "acronym":
            run_acronym(
                each_case, output_file_path, target_generate, 
                attacker_generate, args, integration
            )
        else:
            raise ValueError(f"Invalid jailbreak tactic: {args.attack_method}")
        
        # Log statistics if integration is enabled
        if integration:
            stats = integration.get_statistics()
            print(f"\nCase Statistics:")
            print(f"  Total turns: {stats['total_turns']}")
            print(f"  Current η: {stats['current_eta']:.6f}")
            print(f"  Circuit breaker triggered: {stats['circuit_breaker_triggered']}")
            print(f"  Steering active: {stats['steering_active']}")
            
            # Write stats to output
            with open(output_file_path, 'a') as f:
                f.write(json.dumps({
                    "case_stats": stats,
                    "case_index": i
                }) + '\n')
                
            # Early stopping if circuit breaker triggered (saves compute)
            if stats['circuit_breaker_triggered']:
                print(f"  Circuit breaker triggered - attack stopped early")
                
            # Early stopping if circuit breaker triggered (saves compute)
            if stats['circuit_breaker_triggered']:
                print(f"  Circuit breaker triggered - attack stopped early")

 
    print("Evaluation Complete!")
    print(f"Results saved to: {output_file_path}")
   
