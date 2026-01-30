"""
Response Generator Module
Generates empathetic responses conditioned on detected emotion.
Two modes:
1. GPT-4 API (production) - requires OPENAI_API_KEY
2. DialoGPT-medium (local fallback) - fine-tuned on EmpatheticDialogues
"""

import os
import json
from typing import Dict, List, Optional, Tuple
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import yaml

# Try to import openai, handle if not installed
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI package not installed. Only DialoGPT mode will work.")


class ResponseGenerator:
    """
    Generates empathetic responses conditioned on detected emotion.
    
    Two modes:
    1. GPT-4 API (production) - requires OPENAI_API_KEY
    2. DialoGPT-medium (local fallback) - fine-tuned on EmpatheticDialogues
    """
    
    def __init__(self, config_path: str = "config/config.yaml", mode: str = "gpt4_api"):
        """
        Initialize response generator.
        
        Args:
            config_path: Path to config YAML
            mode: "gpt4_api" or "dialogpt_local"
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.mode = mode
        self.emotion_tokens = {
            'anger': '<anger>',
            'fear': '<fear>',
            'joy': '<joy>',
            'neutral': '<neutral>',
            'sadness': '<sadness>',
            'surprise': '<surprise>'
        }
        
        if mode == "gpt4_api":
            self._init_gpt4()
        elif mode == "dialogpt_local":
            self._init_dialogpt()
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _init_gpt4(self):
        """Initialize GPT-4 API client."""
        if not OPENAI_AVAILABLE:
            print("⚠️  WARNING: OpenAI package not installed. Falling back to DialoGPT.")
            self.mode = "dialogpt_local"
            self._init_dialogpt()
            return
            
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  WARNING: OPENAI_API_KEY not set. Falling back to DialoGPT.")
            self.mode = "dialogpt_local"
            self._init_dialogpt()
            return
        
        self.openai_client = OpenAI(api_key=api_key)
        self.gpt4_model = self.config.get('response_model', {}).get('gpt4_model', 'gpt-4o')
        print(f"✅ GPT-4o API initialized with model: {self.gpt4_model}")
    
    def _init_dialogpt(self):
        """Initialize DialoGPT-medium model."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "microsoft/DialoGPT-medium"
        
        print(f"Loading DialoGPT from {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        
        # Check if fine-tuned version exists
        finetune_path = "models/response_generation/dialogpt_finetuned"
        if os.path.exists(os.path.join(finetune_path, "pytorch_model.bin")):
            print(f"Loading fine-tuned model from {finetune_path}...")
            self.model.load_state_dict(torch.load(f"{finetune_path}/pytorch_model.bin", map_location=device))
            print("✅ Fine-tuned DialoGPT loaded!")
        else:
            print("✅ Base DialoGPT loaded (no fine-tuning found)")
    
    def generate(
        self, 
        user_input: str, 
        emotion: str, 
        history: List[Tuple[str, str]] = None,
        style: str = "casual"
    ) -> str:
        """
        Generate empathetic response.
        
        Args:
            user_input: User's message
            emotion: Detected emotion (anger, fear, joy, neutral, sadness, surprise)
            history: List of (user_msg, bot_response) tuples
            style: "casual" or "formal"
        
        Returns:
            Generated response
        """
        if history is None:
            history = []
        
        if self.mode == "gpt4_api":
            return self._generate_gpt4(user_input, emotion, history, style)
        else:
            return self._generate_dialogpt(user_input, emotion, history, style)
    
    def _generate_gpt4(
        self, 
        user_input: str, 
        emotion: str, 
        history: List[Tuple[str, str]], 
        style: str
    ) -> str:
        """Generate using GPT-4 API."""
        
        # Build conversation history
        history_text = "\n".join([
            f"User: {msg}\nBot: {resp}" 
            for msg, resp in history[-3:]  # Last 3 turns
        ])
        
        prompt = f"""You are an empathetic, compassionate mental health support chatbot.

DETECTED EMOTION: {emotion.upper()}
COMMUNICATION STYLE: {style.upper()}

CONVERSATION HISTORY:
{history_text if history_text else "(This is the start of the conversation)"}

USER'S CURRENT MESSAGE: "{user_input}"

INSTRUCTIONS:
- Acknowledge their {emotion} emotion specifically
- Validate their feelings (DON'T minimize or dismiss)
- Show you understand (use reflective listening)
- Offer gentle support or ask clarifying questions
- DO NOT give medical/psychiatric advice
- DO NOT diagnose
- Keep response to 2-3 sentences (under 100 words)
- Match their communication style ({style})
- End by letting them know you're listening

IMPORTANT SAFETY REMINDERS:
- This is a support chatbot, not professional mental health care
- If they express crisis risk, provide hotline numbers
- Encourage professional help when appropriate

RESPOND AS THE BOT:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.gpt4_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"⚠️  GPT-4 API error: {e}")
            print("Falling back to DialoGPT...")
            self.mode = "dialogpt_local"
            self._init_dialogpt()
            return self._generate_dialogpt(user_input, emotion, history, style)
    
    def _generate_dialogpt(
        self, 
        user_input: str, 
        emotion: str, 
        history: List[Tuple[str, str]], 
        style: str
    ) -> str:
        """Generate using fine-tuned DialoGPT."""
        
        # Add emotion token to input
        emotion_token = self.emotion_tokens.get(emotion, '<neutral>')
        
        # Build dialogue context
        input_text = emotion_token
        for user_msg, bot_resp in history[-2:]:  # Last 2 turns
            input_text += f" {user_msg} {bot_resp}"
        input_text += f" {user_input}"
        
        # Tokenize
        input_ids = self.tokenizer.encode(
            input_text[-256:],  # Limit to prevent overflow
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=input_ids.shape[-1] + 60,
                num_beams=3,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(
            output_ids[:, input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        
        # Clean response
        response = response.strip()
        
        # Ensure response is reasonable length
        if len(response) < 20:
            response = f"I understand you're feeling {emotion}. I'm here to listen. Tell me more about what you're experiencing."
        elif len(response) > 200:
            response = response[:200].rsplit(' ', 1)[0] + "..."
        
        return response
    
    def switch_mode(self, new_mode: str):
        """Switch between GPT-4 API and DialoGPT."""
        if new_mode not in ["gpt4_api", "dialogpt_local"]:
            raise ValueError(f"Unknown mode: {new_mode}")
        
        self.mode = new_mode
        if new_mode == "dialogpt_local":
            self._init_dialogpt()
        elif new_mode == "gpt4_api":
            self._init_gpt4()
        print(f"✅ Switched to {new_mode}")


if __name__ == "__main__":
    # Simple test
    print("Testing Response Generator...")
    
    # Test with DialoGPT (no API key needed)
    gen = ResponseGenerator(mode="dialogpt_local")
    response = gen.generate(
        user_input="I've been feeling really anxious lately",
        emotion="fear",
        history=[],
        style="casual"
    )
    print(f"\nDialoGPT Response:\n{response}")
