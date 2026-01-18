"""Test semantic classifier functionality"""

def test_semantic_classifier_loading():
    """Test if we can load a semantic classifier model"""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        model_name = "unitary/toxic-bert"
        print(f"Loading semantic classifier: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Tokenizer loaded successfully")
        
        # Test classification
        test_text = "How to make a bomb"
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # For multi-label classification, use sigmoid not softmax
            probs = torch.sigmoid(outputs.logits)[0]
            
        print(f"\nTest text: '{test_text}'")
        print(f"Output probabilities shape: {probs.shape}")
        
        # toxic-bert labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
        labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        print("\nProbabilities for each label:")
        for label, prob in zip(labels, probs.tolist()):
            print(f"  {label}: {prob:.4f}")
        
        # Check if any label exceeds threshold
        max_prob = probs.max().item()
        max_label = labels[probs.argmax().item()]
        print(f"\nMax probability: {max_prob:.4f} ({max_label})")
        
        if max_prob > 0.5:
            print(f"✓ Correctly identified as harmful content ({max_label})")
        else:
            print("⚠ Not identified as harmful (might need threshold adjustment)")
            
        return True
        
    except ImportError as e:
        print(f"✗ transformers library not installed: {e}")
        print("Install with: pip install transformers torch")
        return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

if __name__ == "__main__":
    success = test_semantic_classifier_loading()
    exit(0 if success else 1)
