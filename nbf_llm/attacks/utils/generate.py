import json,time,torch
from transformers import AutoTokenizer, AutoModelForCausalLM



def local_phi(model, messages,temperature):
    for _ in range(10):
        try:
            if "lora" in model:
                model_id = "PATH_TO_FINETUNED_PHI4"
            else:
                model_id = "microsoft/phi-4"
            
            if not 'model_hg' in globals():
                global tokenizer
                global model_hg
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model_hg = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model_hg.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model_hg.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            return tokenizer.decode(response, skip_special_tokens=True)
        except Exception as e:
                print("Error in generate: ", e)
                time.sleep(1)
        return "I cannot respond"

def local_llama(model, messages,temperature):
    for _ in range(10):
        try:
            if "lora" in model:
                model_id = "PATH_TO_FINETUNED_LLAMA3"
            else:
                model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

            if not 'model_hg' in globals():
                global tokenizer
                global model_hg
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model_hg = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model_hg.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model_hg.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            return tokenizer.decode(response, skip_special_tokens=True)
        except Exception as e:
                print("Error in generate: ", e)
                time.sleep(1)
        return "I cannot respond"


def generate(messages, client, model, json_format=False, temperature=0.7):
    # Aggressive context trimming to prevent overflow
    def estimate_tokens(text):
        """Conservative token estimation: ~2.5 chars per token"""
        return len(str(text)) // 2  # More conservative estimate
    
    def trim_context(msgs, max_tokens=1500):  # Extremely aggressive - less than half context window
        """Keep system message and recent conversation within very strict token limit"""
        if not msgs:
            return msgs
            
        # Separate system and other messages
        system_msgs = [msg for msg in msgs if msg.get('role') == 'system']
        other_msgs = [msg for msg in msgs if msg.get('role') != 'system']
        
        # Aggressively truncate ALL messages to prevent any single message from being too long
        for msg in system_msgs:
            content = str(msg.get('content', ''))
            if len(content) > 500:  # Very short system messages
                msg['content'] = content[:500] + "..."
                
        for msg in other_msgs:
            content = str(msg.get('content', ''))
            if len(content) > 800:  # Short individual messages
                msg['content'] = content[:800] + "... [truncated]"
        
        # Recalculate tokens after truncation
        current_tokens = sum(estimate_tokens(msg.get('content', '')) for msg in system_msgs)
        
        # Only keep the very last message pair (user+assistant) if it exists
        result_msgs = system_msgs[:]
        if other_msgs:
            # Just take the last user message
            last_msg = other_msgs[-1]
            msg_tokens = estimate_tokens(last_msg.get('content', ''))
            if current_tokens + msg_tokens <= max_tokens:
                result_msgs.append(last_msg)
            
        return result_msgs
    
    # Apply context trimming
    messages = trim_context(messages)
    
    # Additional safety check - ensure total character count is reasonable
    total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
    if total_chars > 3000:  # Very conservative character limit
        print(f"Warning: Message still too long ({total_chars} chars), applying emergency trimming...")
        # Keep only system message and create a minimal user message
        system_msgs = [msg for msg in messages if msg.get('role') == 'system']
        if system_msgs:
            sys_content = str(system_msgs[0].get('content', ''))[:200]
            system_msgs[0]['content'] = sys_content
        messages = system_msgs + [{'role': 'user', 'content': 'Please help with the previous request.'}]
    
    # LM Studio local models (e.g., any GGUF models)
    if hasattr(client, 'base_url') and client.base_url and 'localhost:1234' in str(client.base_url):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            # Try to parse JSON if requested
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
                        return content
            return content
        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['context', 'overflow', 'length', 'token']):
                print(f"Context overflow detected, trying with ultra-minimal context...")
                # Ultra-minimal: Just system message and very short user prompt
                system_msgs = [msg for msg in messages if msg.get('role') == 'system']
                
                # Truncate system message too if needed
                if system_msgs:
                    sys_content = str(system_msgs[0].get('content', ''))
                    if len(sys_content) > 200:
                        system_msgs[0]['content'] = sys_content[:200] + "..."
                
                # Create minimal user message
                minimal_msgs = system_msgs + [{'role': 'user', 'content': 'Please respond briefly to the request.'}]
                    
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=minimal_msgs,
                        temperature=temperature,
                        max_tokens=100,  # Limit response length too
                    )
                    return response.choices[0].message.content
                except:
                    # Absolute last resort - minimal system prompt only
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{'role': 'user', 'content': 'Hello.'}],
                            temperature=temperature,
                            max_tokens=50,
                        )
                        return "Context overflow - minimal response: " + response.choices[0].message.content
                    except:
                        pass
                return "Context too long - cannot respond."
            print(f"LM Studio error: {e}")
            return "I cannot respond"
    # Ollama local models (e.g., llama3.2, mistral, qwen, etc.)
    elif hasattr(client, 'base_url') and client.base_url and 'localhost:11434' in str(client.base_url):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            # Try to parse JSON if requested
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
                        return content
            return content
        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['context', 'overflow', 'length', 'token']):
                print(f"Ollama context overflow detected, trying with minimal context...")
                # Emergency trimming - keep only system message and truncated last message
                system_msgs = [msg for msg in messages if msg.get('role') == 'system']
                other_msgs = [msg for msg in messages if msg.get('role') != 'system']
                
                if other_msgs:
                    last_msg = other_msgs[-1].copy()
                    content = str(last_msg.get('content', ''))
                    last_msg['content'] = content[:300]  # Severe truncation
                    minimal_msgs = system_msgs + [last_msg]
                else:
                    minimal_msgs = system_msgs
                    
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=minimal_msgs,
                        temperature=temperature,
                    )
                    return response.choices[0].message.content
                except:
                    return "Context too long - cannot respond."
            print(f"Ollama error: {e}")
            return "I cannot respond"
    elif "llama-3-8b-instruct" in model:# and json_format:
        response = local_llama(model, messages, temperature)
        return response
    elif "phi" in model:# and json_format:
        assert not json_format
        response = local_phi(model, messages, temperature)
        return response
    elif "claude" in model:
        messages_nosys = [message for message in messages if message['role']!= 'system']
        response = client.messages.create(
        model=model,
        system=[message["content"] for message in messages if message['role']== 'system'][0],
        max_tokens=8192,
        messages=messages_nosys,
        temperature=temperature,
        )
        return response.content[0].text
    elif "llama" in model or "deepseek" in model:
        # use llama-api
        for ind_ in range(100):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    )
                return response.choices[0].message.content
            except Exception as e:
                print("Error in generate: ", e)
                print("generate: ", response.choices[0].message.content)
        return {}
    else:
        # openai gpt 
        for ind_ in range(10):
            try:
                if 'o1' in model or 'o3' in model:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages
                    )
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        response_format={"type": "text"} if not json_format else {"type": "json_object"}
                    )

                if json_format:
                    return json.loads(response.choices[0].message.content)

                return response.choices[0].message.content
            except Exception as e:
                print("Error in generate: ", e)
                print("generate: ", response.choices[0].message.content)
                time.sleep(2*ind_+1)
        return {}
