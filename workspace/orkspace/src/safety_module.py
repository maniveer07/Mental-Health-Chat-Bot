"""
Safety Module for Mental Health Chatbot

This module implements crisis detection, content filtering, and safety mechanisms
to ensure user safety and appropriate responses.
"""

import re
from typing import Dict, List, Tuple, Optional
import logging
from textblob import TextBlob

logger = logging.getLogger(__name__)


class SafetyModule:
    """
    Comprehensive safety module for mental health chatbot.
    Handles crisis detection, content filtering, and resource provision.
    """
    
    # Crisis-related keywords organized by severity
    CRISIS_KEYWORDS = {
        'high': [
            'kill myself', 'end my life', 'suicide', 'suicidal',
            'end it all', 'overdose', 'hurt myself', 'self-harm',
            'cutting myself', 'want to die', 'better off dead',
            'no reason to live', 'goodbye forever', 'final goodbye'
        ],
        'medium': [
            'can\'t go on', 'give up', 'no hope', 'hopeless',
            'worthless', 'don\'t want to live', 'nothing matters',
            'tired of living', 'want it to end', 'can\'t take it anymore'
        ],
        'low': [
            'depressed', 'very sad', 'feeling down', 'lost',
            'alone', 'nobody cares', 'feeling empty'
        ]
    }
    
    # Harmful response patterns to filter
    HARMFUL_PATTERNS = [
        r'kill yourself',
        r'you should die',
        r'end (your|it all)',
        r'nobody cares about you',
        r'you\'re worthless',
        r'give up',
        r'there\'s no hope',
        r'it\'s not worth it'
    ]
    
    # Crisis resources
    CRISIS_RESOURCES = {
        'us': {
            'name': 'National Suicide Prevention Lifeline',
            'phone': '988',
            'description': '24/7 free and confidential support'
        },
        'us_text': {
            'name': 'Crisis Text Line',
            'phone': '741741',
            'description': 'Text HOME to 741741'
        },
        'us_veterans': {
            'name': 'Veterans Crisis Line',
            'phone': '988 (Press 1)',
            'description': 'For veterans and their families'
        },
        'international': {
            'name': 'International Association for Suicide Prevention',
            'url': 'https://www.iasp.info/resources/Crisis_Centres/',
            'description': 'Find crisis centers worldwide'
        },
        'emergency': {
            'name': 'Emergency Services',
            'phone': '911',
            'description': 'For immediate danger'
        }
    }
    
    def __init__(self, config: Dict = None):
        """
        Initialize SafetyModule.
        
        Args:
            config: Configuration dictionary with safety settings
        """
        self.config = config or {}
        self.enable_crisis_detection = self.config.get('safety', {}).get(
            'enable_crisis_detection', True
        )
        self.provide_resources = self.config.get('safety', {}).get(
            'provide_resources', True
        )
        self.escalation_threshold = self.config.get('safety', {}).get(
            'escalation_threshold', 0.8
        )
        
        # Load custom keywords if provided
        if config and 'safety' in config and 'crisis_keywords' in config['safety']:
            custom_keywords = config['safety']['crisis_keywords']
            if isinstance(custom_keywords, list):
                self.CRISIS_KEYWORDS['high'].extend(custom_keywords)
        
        logger.info("SafetyModule initialized")
    
    def detect_crisis_keywords(self, text: str) -> Dict[str, any]:
        """
        Detect crisis-related keywords in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detection results
        """
        if not self.enable_crisis_detection:
            return {'is_crisis': False, 'severity': 'none', 'matched_keywords': []}
        
        text_lower = text.lower()
        matched_keywords = []
        severity = 'none'
        
        # Check high severity keywords first
        for keyword in self.CRISIS_KEYWORDS['high']:
            if keyword in text_lower:
                matched_keywords.append(keyword)
                severity = 'high'
        
        # Check medium severity if no high severity found
        if severity == 'none':
            for keyword in self.CRISIS_KEYWORDS['medium']:
                if keyword in text_lower:
                    matched_keywords.append(keyword)
                    severity = 'medium'
        
        # Check low severity if no medium severity found
        if severity == 'none':
            for keyword in self.CRISIS_KEYWORDS['low']:
                if keyword in text_lower:
                    matched_keywords.append(keyword)
                    severity = 'low'
        
        is_crisis = severity in ['high', 'medium']
        
        if is_crisis:
            logger.warning(f"Crisis detected - Severity: {severity}, Keywords: {matched_keywords}")
        
        return {
            'is_crisis': is_crisis,
            'severity': severity,
            'matched_keywords': matched_keywords,
            'requires_immediate_resources': severity == 'high'
        }
    
    def assess_risk_level(
        self,
        text: str,
        emotion: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, any]:
        """
        Comprehensive risk assessment based on text, emotion, and conversation history.
        
        Args:
            text: Current message text
            emotion: Detected emotion
            conversation_history: Previous conversation turns
            
        Returns:
            Dictionary with risk assessment
        """
        # Keyword-based crisis detection
        crisis_result = self.detect_crisis_keywords(text)
        
        # Emotion-based risk factors
        high_risk_emotions = ['grief', 'sadness', 'fear', 'remorse', 'disappointment']
        emotion_risk = 0.7 if emotion in high_risk_emotions else 0.3
        
        # Sentiment analysis
        sentiment_score = self._analyze_sentiment(text)
        sentiment_risk = 1.0 - (sentiment_score + 1) / 2  # Convert [-1, 1] to [0, 1]
        
        # Historical pattern analysis
        history_risk = 0.0
        if conversation_history:
            history_risk = self._analyze_conversation_history(conversation_history)
        
        # Calculate overall risk score (weighted average)
        weights = {
            'keywords': 0.5,
            'emotion': 0.2,
            'sentiment': 0.2,
            'history': 0.1
        }
        
        keyword_risk = {
            'high': 1.0,
            'medium': 0.7,
            'low': 0.4,
            'none': 0.0
        }[crisis_result['severity']]
        
        overall_risk = (
            weights['keywords'] * keyword_risk +
            weights['emotion'] * emotion_risk +
            weights['sentiment'] * sentiment_risk +
            weights['history'] * history_risk
        )
        
        # Determine risk level
        if overall_risk >= 0.8:
            risk_level = 'critical'
        elif overall_risk >= 0.6:
            risk_level = 'high'
        elif overall_risk >= 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'overall_risk_score': overall_risk,
            'risk_level': risk_level,
            'crisis_detected': crisis_result['is_crisis'],
            'severity': crisis_result['severity'],
            'matched_keywords': crisis_result['matched_keywords'],
            'requires_intervention': overall_risk >= self.escalation_threshold,
            'provide_resources': crisis_result['is_crisis'] or overall_risk >= 0.6
        }
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using TextBlob.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment polarity score (-1 to 1)
        """
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def _analyze_conversation_history(
        self,
        conversation_history: List[Dict[str, str]]
    ) -> float:
        """
        Analyze conversation history for escalating distress patterns.
        
        Args:
            conversation_history: List of conversation turns
            
        Returns:
            Risk score from 0 to 1
        """
        if not conversation_history or len(conversation_history) < 2:
            return 0.0
        
        # Look at recent messages (last 5)
        recent_messages = conversation_history[-5:]
        
        # Count crisis keywords over time
        crisis_count = 0
        for turn in recent_messages:
            if 'content' in turn:
                result = self.detect_crisis_keywords(turn['content'])
                if result['is_crisis']:
                    crisis_count += 1
        
        # Calculate risk based on frequency
        risk = min(crisis_count / len(recent_messages), 1.0)
        
        return risk
    
    def get_crisis_resources(
        self,
        severity: str = 'high',
        format_type: str = 'text'
    ) -> str:
        """
        Get formatted crisis resources based on severity.
        
        Args:
            severity: Crisis severity level
            format_type: Format type ('text', 'markdown', 'html')
            
        Returns:
            Formatted crisis resources string
        """
        if not self.provide_resources:
            return ""
        
        if format_type == 'markdown':
            return self._format_resources_markdown(severity)
        elif format_type == 'html':
            return self._format_resources_html(severity)
        else:
            return self._format_resources_text(severity)
    
    def _format_resources_text(self, severity: str) -> str:
        """Format crisis resources as plain text."""
        resources = []
        
        if severity in ['high', 'critical']:
            resources.append("\n🆘 IMMEDIATE HELP AVAILABLE 🆘\n")
            resources.append("If you're in immediate danger, please call 911.\n")
        
        resources.append("📞 Crisis Support Resources:\n")
        
        for key, resource in self.CRISIS_RESOURCES.items():
            if 'phone' in resource:
                resources.append(f"• {resource['name']}: {resource['phone']}")
                resources.append(f"  {resource['description']}\n")
            elif 'url' in resource:
                resources.append(f"• {resource['name']}")
                resources.append(f"  {resource['url']}")
                resources.append(f"  {resource['description']}\n")
        
        resources.append("\nYou are not alone. Help is available 24/7.")
        
        return '\n'.join(resources)
    
    def _format_resources_markdown(self, severity: str) -> str:
        """Format crisis resources as markdown."""
        resources = []
        
        if severity in ['high', 'critical']:
            resources.append("## 🆘 IMMEDIATE HELP AVAILABLE\n")
            resources.append("**If you're in immediate danger, please call 911.**\n")
        
        resources.append("### 📞 Crisis Support Resources\n")
        
        for key, resource in self.CRISIS_RESOURCES.items():
            if 'phone' in resource:
                resources.append(f"**{resource['name']}**: {resource['phone']}")
                resources.append(f"*{resource['description']}*\n")
            elif 'url' in resource:
                resources.append(f"**{resource['name']}**")
                resources.append(f"[{resource['url']}]({resource['url']})")
                resources.append(f"*{resource['description']}*\n")
        
        resources.append("**You are not alone. Help is available 24/7.**")
        
        return '\n'.join(resources)
    
    def _format_resources_html(self, severity: str) -> str:
        """Format crisis resources as HTML."""
        html = ['<div class="crisis-resources">']
        
        if severity in ['high', 'critical']:
            html.append('<h3 style="color: red;">🆘 IMMEDIATE HELP AVAILABLE</h3>')
            html.append('<p><strong>If you\'re in immediate danger, please call 911.</strong></p>')
        
        html.append('<h4>📞 Crisis Support Resources</h4>')
        html.append('<ul>')
        
        for key, resource in self.CRISIS_RESOURCES.items():
            html.append('<li>')
            html.append(f'<strong>{resource["name"]}</strong>: ')
            if 'phone' in resource:
                html.append(f'<a href="tel:{resource["phone"]}">{resource["phone"]}</a>')
            elif 'url' in resource:
                html.append(f'<a href="{resource["url"]}" target="_blank">Visit Website</a>')
            html.append(f'<br><em>{resource["description"]}</em>')
            html.append('</li>')
        
        html.append('</ul>')
        html.append('<p><strong>You are not alone. Help is available 24/7.</strong></p>')
        html.append('</div>')
        
        return '\n'.join(html)
    
    def validate_response(self, response: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a generated response is safe and appropriate.
        
        Args:
            response: Generated response text
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check for empty response
        if not response or len(response.strip()) == 0:
            return False, "Empty response"
        
        # Check minimum length
        if len(response.split()) < 3:
            return False, "Response too short"
        
        # Check for harmful patterns
        response_lower = response.lower()
        for pattern in self.HARMFUL_PATTERNS:
            if re.search(pattern, response_lower):
                logger.warning(f"Harmful pattern detected in response: {pattern}")
                return False, f"Contains harmful content: {pattern}"
        
        # Check for encouraging self-harm
        harmful_phrases = [
            'kill yourself',
            'end your life',
            'you should die',
            'nobody cares',
            'give up on life',
            'not worth living'
        ]
        
        for phrase in harmful_phrases:
            if phrase in response_lower:
                logger.warning(f"Harmful phrase detected: {phrase}")
                return False, f"Contains harmful phrase: {phrase}"
        
        # Check for inappropriate medical advice
        medical_advice_patterns = [
            r'take \d+ (pills|tablets|medication)',
            r'stop taking (your )?medication',
            r'don\'t need (a )?doctor',
            r'(i|you) can diagnose'
        ]
        
        for pattern in medical_advice_patterns:
            if re.search(pattern, response_lower):
                logger.warning(f"Inappropriate medical advice detected: {pattern}")
                return False, "Contains inappropriate medical advice"
        
        # Sentiment check - response shouldn't be overly negative
        sentiment = self._analyze_sentiment(response)
        if sentiment < -0.5:
            return False, "Response sentiment too negative"
        
        return True, None
    
    def filter_harmful_content(self, text: str) -> str:
        """
        Filter and sanitize potentially harmful content.
        
        Args:
            text: Input text
            
        Returns:
            Sanitized text
        """
        # Remove explicit harmful phrases
        filtered_text = text
        
        for pattern in self.HARMFUL_PATTERNS:
            filtered_text = re.sub(pattern, '[FILTERED]', filtered_text, flags=re.IGNORECASE)
        
        return filtered_text
    
    def should_escalate_to_professional(
        self,
        conversation_history: List[Dict[str, str]],
        risk_score: float
    ) -> Dict[str, any]:
        """
        Determine if conversation should be escalated to a professional.
        
        Args:
            conversation_history: Full conversation history
            risk_score: Current risk assessment score
            
        Returns:
            Dictionary with escalation recommendation
        """
        should_escalate = False
        reasons = []
        
        # Check risk score
        if risk_score >= self.escalation_threshold:
            should_escalate = True
            reasons.append("High risk score detected")
        
        # Check conversation length with persistent distress
        if len(conversation_history) > 10:
            # Count distressing messages
            distress_count = 0
            for turn in conversation_history[-10:]:
                if 'content' in turn:
                    crisis_result = self.detect_crisis_keywords(turn['content'])
                    if crisis_result['severity'] in ['medium', 'high']:
                        distress_count += 1
            
            if distress_count >= 5:
                should_escalate = True
                reasons.append("Persistent distress over multiple turns")
        
        # Check for escalating severity
        if len(conversation_history) >= 3:
            recent_severities = []
            for turn in conversation_history[-3:]:
                if 'content' in turn:
                    result = self.detect_crisis_keywords(turn['content'])
                    severity_score = {'high': 3, 'medium': 2, 'low': 1, 'none': 0}[result['severity']]
                    recent_severities.append(severity_score)
            
            # Check if severity is increasing
            if len(recent_severities) == 3 and recent_severities[-1] > recent_severities[0]:
                should_escalate = True
                reasons.append("Escalating severity detected")
        
        recommendation = {
            'should_escalate': should_escalate,
            'reasons': reasons,
            'message': self._get_escalation_message() if should_escalate else None
        }
        
        if should_escalate:
            logger.warning(f"Escalation recommended: {reasons}")
        
        return recommendation
    
    def _get_escalation_message(self) -> str:
        """Get escalation message for professional referral."""
        return (
            "I notice you're going through a very difficult time. "
            "While I'm here to listen, I think it would be really helpful to speak with "
            "a trained mental health professional who can provide the support you deserve. "
            "Would you consider reaching out to one of the crisis resources I've shared?"
        )
    
    def get_safe_fallback_response(self, emotion: str = "neutral") -> str:
        """
        Get a safe fallback response when generated response fails validation.
        
        Args:
            emotion: Detected emotion
            
        Returns:
            Safe fallback response
        """
        fallback_responses = {
            'sadness': "I hear that you're going through a difficult time. Your feelings are valid, and I'm here to listen. Could you tell me more about what's troubling you?",
            'fear': "It sounds like you're feeling scared or worried. That's completely understandable. I'm here to support you. What's been on your mind?",
            'anger': "I understand you're feeling frustrated. Your emotions are important, and it's okay to express them. Would you like to talk about what's bothering you?",
            'neutral': "I'm here to listen and support you. Could you tell me more about how you're feeling right now?"
        }
        
        return fallback_responses.get(emotion, fallback_responses['neutral'])


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize safety module
    safety = SafetyModule()
    
    # Test crisis detection
    test_messages = [
        "I've been feeling really anxious lately",
        "I can't go on like this anymore",
        "I want to kill myself",
        "Today was a good day",
        "I feel so hopeless and worthless"
    ]
    
    print("\nTesting crisis detection:")
    for msg in test_messages:
        result = safety.assess_risk_level(msg, "sadness")
        print(f"\nMessage: {msg}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Risk Score: {result['overall_risk_score']:.2f}")
        print(f"Crisis Detected: {result['crisis_detected']}")
        
        if result['provide_resources']:
            print("\nCrisis Resources:")
            print(safety.get_crisis_resources(result['severity']))
