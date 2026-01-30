"""
Chatbot Interface Module using Chainlit

This module provides a beautiful chat interface for the emotion-aware mental health chatbot
with real-time emotion detection and empathetic responses.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import chainlit as cl
import yaml
import logging

# Import custom modules
from emotion_detector import EmotionDetector
from response_generator import ResponseGenerator
from safety_module import SafetyModule

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation history and context for the chatbot.
    """
    
    def __init__(self, max_history: int = 50, context_window: int = 5):
        """
        Initialize conversation manager.
        
        Args:
            max_history: Maximum number of messages to store
            context_window: Number of recent messages to use for context
        """
        self.max_history = max_history
        self.context_window = context_window
        self.conversation_history = []
        self.session_start = datetime.now()
    
    def add_message(
        self,
        role: str,
        content: str,
        emotion: Optional[str] = None,
        confidence: Optional[float] = None
    ):
        """
        Add a message to conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            emotion: Detected emotion (for user messages)
            confidence: Emotion confidence score
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'emotion': emotion,
            'confidence': confidence
        }
        
        self.conversation_history.append(message)
        
        # Trim history if exceeds max length
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_context(self, num_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get recent conversation context.
        
        Args:
            num_messages: Number of recent messages (uses context_window if None)
            
        Returns:
            List of recent messages
        """
        if num_messages is None:
            num_messages = self.context_window
        
        return self.conversation_history[-num_messages:]
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.session_start = datetime.now()
        logger.info("Conversation history cleared")
    
    def export_conversation(self, format: str = 'txt') -> str:
        """
        Export conversation history.
        
        Args:
            format: Export format ('json', 'txt')
            
        Returns:
            Exported conversation string
        """
        if format == 'json':
            return json.dumps(self.conversation_history, indent=2)
        elif format == 'txt':
            lines = []
            lines.append(f"Conversation Log - Started: {self.session_start.isoformat()}\n")
            lines.append("=" * 60 + "\n")
            
            for msg in self.conversation_history:
                timestamp = msg['timestamp']
                role = msg['role'].capitalize()
                content = msg['content']
                
                lines.append(f"[{timestamp}] {role}:\n{content}\n")
                
                if msg.get('emotion'):
                    lines.append(f"  (Emotion: {msg['emotion']}, Confidence: {msg.get('confidence', 0):.2f})\n")
                
                lines.append("\n")
            
            return ''.join(lines)
        else:
            return str(self.conversation_history)


# Emoji mapping for emotions
EMOTION_EMOJIS = {
    'anger': '😠',
    'fear': '😨',
    'joy': '😊',
    'neutral': '😐',
    'sadness': '😢',
    'surprise': '😮'
}


@cl.on_chat_start
async def start():
    """
    Initialize chatbot when a new chat session starts.
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/chatbot_chainlit.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Starting new chat session")
    
    # Load configuration
    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    await cl.Message(content="🔄 Initializing chatbot... Please wait a moment.").send()
    
    # Emotion detector
    emotion_detector = EmotionDetector(
        model_name=config['model']['emotion_detection']['model_name'],
        num_labels=config['model']['emotion_detection']['num_labels'],
        max_length=config['model']['emotion_detection']['max_length']
    )
    
    # Load trained model if available
    emotion_model_path = os.path.join(
        config['paths']['checkpoints_dir'],
        'best_model'
    )
    
    if os.path.exists(emotion_model_path):
        try:
            emotion_detector.load_model(emotion_model_path)
            logger.info(f"Loaded trained emotion detection model from {emotion_model_path}")
        except Exception as e:
            logger.warning(f"Could not load trained emotion model: {e}")
    
    # Response generator (GPT-4 API or DialoGPT)
    response_mode = os.getenv("RESPONSE_MODE", "gpt4_api")
    response_generator = ResponseGenerator(
        config_path="config/config.yaml",
        mode=response_mode
    )
    
    # Safety module
    safety_module = SafetyModule(config=config)
    
    # Conversation manager
    conversation_manager = ConversationManager(
        max_history=config['interface']['max_conversation_length'],
        context_window=config['interface']['context_window']
    )
    
    # Store in user session
    cl.user_session.set("emotion_detector", emotion_detector)
    cl.user_session.set("response_generator", response_generator)
    cl.user_session.set("safety_module", safety_module)
    cl.user_session.set("conversation_manager", conversation_manager)
    cl.user_session.set("config", config)
    
    # Send welcome message
    mode_name = "GPT-4o API" if response_mode == "gpt4_api" else "Local DialoGPT"
    
    welcome_message = f"""# 🧠 Welcome to the Emotion-Aware Mental Health Chatbot

I'm here to listen and provide empathetic support. I use advanced AI to understand your emotions and respond appropriately.

**Current Mode:** {mode_name}  
**Emotion Detection:** RoBERTa-base with LoRA (6 emotions)

---

### ⚠️ Important Disclaimer

{config['interface']['disclaimer']}

---

### 💡 How to Use:
- Share your thoughts and feelings freely
- I'll detect your emotion and respond with empathy
- If you're in crisis, I'll provide emergency resources
- Your conversations are private and not permanently stored

### 🎯 Example Messages to Try:
- "I'm feeling really anxious about my exam tomorrow"
- "I got accepted into my dream university!"
- "I've been feeling really sad and lonely lately"

**Ready to chat? Type your message below!** 👇
"""
    
    await cl.Message(content=welcome_message).send()
    logger.info("Chatbot initialized successfully")


@cl.on_message
async def main(message: cl.Message):
    """
    Process incoming user messages.
    
    Args:
        message: User's message from Chainlit
    """
    # Get components from session
    emotion_detector = cl.user_session.get("emotion_detector")
    response_generator = cl.user_session.get("response_generator")
    safety_module = cl.user_session.get("safety_module")
    conversation_manager = cl.user_session.get("conversation_manager")
    config = cl.user_session.get("config")
    
    user_input = message.content
    
    # Show processing message
    processing_msg = await cl.Message(content="🔍 Analyzing your emotion...").send()
    
    try:
        # Step 1: Detect emotion
        emotion_result = emotion_detector.predict_emotion(
            user_input,
            return_probabilities=True
        )
        
        detected_emotion = emotion_result['emotion']
        confidence = emotion_result['confidence']
        
        # Add user message to conversation manager
        conversation_manager.add_message(
            role='user',
            content=user_input,
            emotion=detected_emotion,
            confidence=confidence
        )
        
        # Step 2: Assess safety/risk
        processing_msg.content = "🛡️ Checking for safety concerns..."
        await processing_msg.update()
        
        risk_assessment = safety_module.assess_risk_level(
            text=user_input,
            emotion=detected_emotion,
            conversation_history=conversation_manager.get_context()
        )
        
        # Step 3: Generate response
        processing_msg.content = "💭 Generating response..."
        await processing_msg.update()
        
        # Handle crisis situation
        if risk_assessment['crisis_detected'] or risk_assessment['provide_resources']:
            logger.warning(f"Crisis detected - Risk level: {risk_assessment['risk_level']}")
            
            response = _generate_crisis_response(risk_assessment, safety_module)
            
            # Add to conversation manager
            conversation_manager.add_message(
                role='assistant',
                content=response
            )
            
            # Send crisis response with warning
            await processing_msg.remove()
            
            # Send emotion info with visual chart
            emotion_emoji = EMOTION_EMOJIS.get(detected_emotion, '🤔')
            emotion_text = f"{emotion_emoji} **Detected Emotion:** {detected_emotion.title()} ({confidence:.1%} confidence)"
            
            if 'top_emotions' in emotion_result:
                top_3 = emotion_result['top_emotions'][:3]
                emotion_details = "\n".join([
                    f"  - {EMOTION_EMOJIS.get(e['emotion'], '🤔')} {e['emotion'].title()}: {e['probability']:.1%}"
                    for e in top_3
                ])
                emotion_text += f"\n\n**Top Emotions:**\n{emotion_details}"
                
                # Create ASCII bar chart for visualization
                emotion_text += "\n\n**Visual Chart:**\n```"
                for e in top_3:
                    bar_length = int(e['probability'] * 30)  # Scale to 30 chars max
                    bar = "█" * bar_length
                    emotion_text += f"\n{e['emotion'].title():10s} {bar} {e['probability']:.1%}"
                emotion_text += "\n```"
            
            await cl.Message(
                content=emotion_text,
                author="Emotion Analyzer"
            ).send()
            
            # Send crisis warning
            if risk_assessment['severity'] in ['high', 'critical']:
                await cl.Message(
                    content="⚠️ **CRISIS DETECTED** - Please see emergency resources below",
                    author="Safety Monitor"
                ).send()
            
            # Send main response
            await cl.Message(content=response).send()
            
            return
        
        # Generate normal empathetic response
        # Convert conversation history to list of tuples
        history_tuples = []
        context = conversation_manager.get_context()
        for i in range(0, len(context)-1, 2):
            if i+1 < len(context):
                user_msg = context[i]['content'] if context[i]['role'] == 'user' else ''
                bot_resp = context[i+1]['content'] if context[i+1]['role'] == 'assistant' else ''
                if user_msg and bot_resp:
                    history_tuples.append((user_msg, bot_resp))
        
        response = response_generator.generate(
            user_input=user_input,
            emotion=detected_emotion,
            history=history_tuples,
            style="casual"
        )
        
        # Validate response safety
        is_valid, reason = safety_module.validate_response(response)
        
        if not is_valid:
            logger.warning(f"Generated response failed validation: {reason}")
            response = safety_module.get_safe_fallback_response(detected_emotion)
        
        # Add response to conversation manager
        conversation_manager.add_message(
            role='assistant',
            content=response
        )
        
        # Check for professional escalation
        escalation = safety_module.should_escalate_to_professional(
            conversation_history=conversation_manager.conversation_history,
            risk_score=risk_assessment['overall_risk_score']
        )
        
        if escalation['should_escalate']:
            response += f"\n\n{escalation['message']}"
        
        # Remove processing message
        await processing_msg.remove()
        
        # Send emotion info with visual chart
        emotion_emoji = EMOTION_EMOJIS.get(detected_emotion, '🤔')
        emotion_text = f"{emotion_emoji} **Detected Emotion:** {detected_emotion.title()} ({confidence:.1%} confidence)"
        
        if 'top_emotions' in emotion_result:
            top_3 = emotion_result['top_emotions'][:3]
            emotion_details = "\n".join([
                f"  - {EMOTION_EMOJIS.get(e['emotion'], '🤔')} {e['emotion'].title()}: {e['probability']:.1%}"
                for e in top_3
            ])
            emotion_text += f"\n\n**Top Emotions:**\n{emotion_details}"
            
            # Create ASCII bar chart for visualization
            emotion_text += "\n\n**Visual Chart:**\n```"
            for e in top_3:
                bar_length = int(e['probability'] * 30)  # Scale to 30 chars max
                bar = "█" * bar_length
                emotion_text += f"\n{e['emotion'].title():10s} {bar} {e['probability']:.1%}"
            emotion_text += "\n```"
        
        await cl.Message(
            content=emotion_text,
            author="Emotion Analyzer"
        ).send()
        
        # Send main response
        await cl.Message(content=response).send()
        
        # Send escalation warning if needed
        if escalation['should_escalate']:
            await cl.Message(
                content="⚠️ **Professional support recommended** - Please consider reaching out to a mental health professional",
                author="Safety Monitor"
            ).send()
    
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await processing_msg.remove()
        
        await cl.Message(
            content=f"I apologize, but I encountered an error processing your message. Please try again. If the issue persists, please restart the chat.\n\nError: {str(e)}"
        ).send()


def _generate_crisis_response(risk_assessment: Dict, safety_module) -> str:
    """
    Generate appropriate response for crisis situations.
    
    Args:
        risk_assessment: Risk assessment results
        safety_module: Safety module instance
        
    Returns:
        Crisis response text
    """
    severity = risk_assessment['severity']
    
    if severity == 'high' or risk_assessment['requires_immediate_resources']:
        response = (
            "I'm really concerned about what you've shared. Your safety is the most important thing. "
            "Please reach out to a crisis counselor right away - they're trained to help and available 24/7.\n\n"
        )
        response += safety_module.get_crisis_resources(severity, format_type='text')
    else:
        response = (
            "I hear that you're going through a very difficult time. It's important to talk to someone "
            "who can provide professional support. Here are some resources that might help:\n\n"
        )
        response += safety_module.get_crisis_resources(severity, format_type='text')
    
    return response


@cl.on_chat_end
def end():
    """
    Clean up when chat session ends.
    """
    logger.info("Chat session ended")


# Optional: Add action buttons for exporting conversation, clearing history, etc.
@cl.action_callback("export_conversation")
async def on_action(action):
    """
    Handle action button clicks.
    """
    conversation_manager = cl.user_session.get("conversation_manager")
    
    if action.name == "export_conversation":
        exported = conversation_manager.export_conversation(format='txt')
        
        # Send as a text file
        await cl.Message(
            content="Here's your conversation history:",
            elements=[
                cl.Text(
                    name="conversation_history.txt",
                    content=exported,
                    display="inline"
                )
            ]
        ).send()

