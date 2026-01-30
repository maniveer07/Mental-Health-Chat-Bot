"""
Unit Tests for Mental Health Chatbot

This module contains unit tests for all major components of the chatbot system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import yaml
import pandas as pd
from unittest.mock import Mock, patch


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader module."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            cls.config = yaml.safe_load(f)
        
        from data_loader_multi import MultiDatasetLoader
        cls.data_loader = MultiDatasetLoader(cls.config)
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        self.assertIsNotNone(self.data_loader)
        self.assertEqual(len(self.data_loader.GOEMOTIONS_EMOTIONS), 28)
        self.assertIn('sadness', self.data_loader.GOEMOTIONS_EMOTIONS)
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        # Test whitespace removal
        text = "  This   has   extra   spaces  "
        processed = self.data_loader.preprocess_text(text)
        self.assertEqual(processed, "This has extra spaces")
        
        # Test empty string
        self.assertEqual(self.data_loader.preprocess_text(""), "")
        
        # Test None handling
        self.assertEqual(self.data_loader.preprocess_text(None), "")
    
    def test_emotion_mappings(self):
        """Test emotion to ID mappings."""
        self.assertEqual(len(self.data_loader.emotion_to_id), 28)
        self.assertEqual(len(self.data_loader.id_to_emotion), 28)
        
        # Test bidirectional mapping
        for emotion, idx in self.data_loader.emotion_to_id.items():
            self.assertEqual(self.data_loader.id_to_emotion[idx], emotion)


class TestEmotionDetector(unittest.TestCase):
    """Test cases for EmotionDetector module."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        from emotion_detector import EmotionDetector
        cls.detector = EmotionDetector(num_labels=28, max_length=128)
    
    def test_initialization(self):
        """Test EmotionDetector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.tokenizer)
        self.assertEqual(self.detector.num_labels, 28)
        self.assertEqual(self.detector.max_length, 128)
    
    def test_preprocess_input(self):
        """Test input preprocessing."""
        text = "I am feeling happy today"
        processed = self.detector.preprocess_input(text)
        
        self.assertIn('input_ids', processed)
        self.assertIn('attention_mask', processed)
        self.assertEqual(processed['input_ids'].shape[1], self.detector.max_length)
    
    def test_predict_emotion(self):
        """Test emotion prediction."""
        text = "I am feeling happy today"
        result = self.detector.predict_emotion(text)
        
        self.assertIn('emotion', result)
        self.assertIn('emotion_id', result)
        self.assertIn('confidence', result)
        self.assertIsInstance(result['emotion'], str)
        self.assertIsInstance(result['confidence'], float)
        self.assertGreater(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 1)
    
    def test_predict_emotion_with_probabilities(self):
        """Test emotion prediction with probability distribution."""
        text = "I am feeling sad"
        result = self.detector.predict_emotion(text, return_probabilities=True)
        
        self.assertIn('top_emotions', result)
        self.assertEqual(len(result['top_emotions']), 5)
        
        for emotion_data in result['top_emotions']:
            self.assertIn('emotion', emotion_data)
            self.assertIn('probability', emotion_data)
    
    def test_predict_batch(self):
        """Test batch prediction."""
        texts = [
            "I am happy",
            "I am sad",
            "I am angry"
        ]
        
        results = self.detector.predict_batch(texts, batch_size=2)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('emotion', result)
            self.assertIn('confidence', result)


class TestSafetyModule(unittest.TestCase):
    """Test cases for SafetyModule."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        from safety_module import SafetyModule
        cls.safety = SafetyModule()
    
    def test_initialization(self):
        """Test SafetyModule initialization."""
        self.assertIsNotNone(self.safety)
        self.assertTrue(self.safety.enable_crisis_detection)
        self.assertTrue(self.safety.provide_resources)
    
    def test_crisis_detection_high_severity(self):
        """Test detection of high severity crisis keywords."""
        text = "I want to kill myself"
        result = self.safety.detect_crisis_keywords(text)
        
        self.assertTrue(result['is_crisis'])
        self.assertEqual(result['severity'], 'high')
        self.assertGreater(len(result['matched_keywords']), 0)
        self.assertTrue(result['requires_immediate_resources'])
    
    def test_crisis_detection_medium_severity(self):
        """Test detection of medium severity distress."""
        text = "I feel hopeless and can't go on"
        result = self.safety.detect_crisis_keywords(text)
        
        self.assertTrue(result['is_crisis'])
        self.assertIn(result['severity'], ['medium', 'high'])
    
    def test_crisis_detection_no_crisis(self):
        """Test normal message without crisis indicators."""
        text = "I had a good day today"
        result = self.safety.detect_crisis_keywords(text)
        
        self.assertFalse(result['is_crisis'])
        self.assertEqual(len(result['matched_keywords']), 0)
    
    def test_assess_risk_level(self):
        """Test comprehensive risk assessment."""
        text = "I can't take this anymore"
        emotion = "sadness"
        
        result = self.safety.assess_risk_level(text, emotion)
        
        self.assertIn('overall_risk_score', result)
        self.assertIn('risk_level', result)
        self.assertIn('crisis_detected', result)
        self.assertGreaterEqual(result['overall_risk_score'], 0)
        self.assertLessEqual(result['overall_risk_score'], 1)
    
    def test_validate_response_safe(self):
        """Test validation of safe responses."""
        safe_response = "I understand you're going through a difficult time. I'm here to listen."
        is_valid, reason = self.safety.validate_response(safe_response)
        
        self.assertTrue(is_valid)
        self.assertIsNone(reason)
    
    def test_validate_response_harmful(self):
        """Test detection of harmful responses."""
        harmful_response = "You should just give up"
        is_valid, reason = self.safety.validate_response(harmful_response)
        
        self.assertFalse(is_valid)
        self.assertIsNotNone(reason)
    
    def test_validate_response_empty(self):
        """Test validation of empty responses."""
        is_valid, reason = self.safety.validate_response("")
        
        self.assertFalse(is_valid)
        self.assertEqual(reason, "Empty response")
    
    def test_get_crisis_resources(self):
        """Test crisis resource retrieval."""
        resources = self.safety.get_crisis_resources(severity='high')
        
        self.assertIsInstance(resources, str)
        self.assertIn('988', resources)  # US crisis line
        self.assertGreater(len(resources), 0)
    
    def test_should_escalate_to_professional(self):
        """Test escalation recommendation logic."""
        conversation_history = [
            {'content': 'I feel sad', 'role': 'user'},
            {'content': 'Response', 'role': 'assistant'},
            {'content': 'I feel hopeless', 'role': 'user'},
            {'content': 'Response', 'role': 'assistant'},
            {'content': 'I want to give up', 'role': 'user'}
        ]
        
        result = self.safety.should_escalate_to_professional(
            conversation_history,
            risk_score=0.85
        )
        
        self.assertIn('should_escalate', result)
        self.assertIn('reasons', result)
        self.assertIsInstance(result['should_escalate'], bool)


class TestResponseGenerator(unittest.TestCase):
    """Test cases for ResponseGenerator module."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        from response_generator import ResponseGenerator
        cls.generator = ResponseGenerator(model_name="gpt2", max_length=100)
    
    def test_initialization(self):
        """Test ResponseGenerator initialization."""
        self.assertIsNotNone(self.generator)
        self.assertIsNotNone(self.generator.model)
        self.assertIsNotNone(self.generator.tokenizer)
        self.assertEqual(self.generator.max_length, 100)
    
    def test_generate_response(self):
        """Test basic response generation."""
        message = "I am feeling anxious"
        emotion = "nervousness"
        
        response = self.generator.generate_response(
            user_message=message,
            detected_emotion=emotion,
            temperature=0.7
        )
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_condition_on_emotion(self):
        """Test emotion conditioning."""
        emotion = "sadness"
        context = "I am feeling down"
        
        conditioned = self.generator.condition_on_emotion(emotion, context)
        
        self.assertIn(emotion, conditioned)
        self.assertIn(context, conditioned)
    
    def test_validate_response_safety(self):
        """Test basic response safety validation."""
        safe_response = "I understand how you feel"
        self.assertTrue(self.generator.validate_response_safety(safe_response))
        
        empty_response = ""
        self.assertFalse(self.generator.validate_response_safety(empty_response))


class TestConversationManager(unittest.TestCase):
    """Test cases for ConversationManager."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        from chatbot_interface import ConversationManager
        cls.manager = ConversationManager(max_history=10, context_window=5)
    
    def test_initialization(self):
        """Test ConversationManager initialization."""
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager.max_history, 10)
        self.assertEqual(self.manager.context_window, 5)
        self.assertEqual(len(self.manager.conversation_history), 0)
    
    def test_add_message(self):
        """Test adding messages to history."""
        self.manager.add_message('user', 'Hello', 'neutral', 0.9)
        
        self.assertEqual(len(self.manager.conversation_history), 1)
        
        last_msg = self.manager.conversation_history[-1]
        self.assertEqual(last_msg['role'], 'user')
        self.assertEqual(last_msg['content'], 'Hello')
        self.assertEqual(last_msg['emotion'], 'neutral')
    
    def test_get_context(self):
        """Test retrieving conversation context."""
        # Add multiple messages
        for i in range(10):
            self.manager.add_message('user', f'Message {i}', 'neutral')
        
        context = self.manager.get_context(num_messages=3)
        
        self.assertEqual(len(context), 3)
        self.assertEqual(context[-1]['content'], 'Message 9')
    
    def test_clear_history(self):
        """Test clearing conversation history."""
        self.manager.add_message('user', 'Test', 'neutral')
        self.manager.clear_history()
        
        self.assertEqual(len(self.manager.conversation_history), 0)
    
    def test_export_conversation(self):
        """Test conversation export."""
        self.manager.clear_history()
        self.manager.add_message('user', 'Hello', 'joy', 0.9)
        self.manager.add_message('assistant', 'Hi there!', None, None)
        
        # Test JSON export
        json_export = self.manager.export_conversation(format='json')
        self.assertIn('Hello', json_export)
        
        # Test text export
        text_export = self.manager.export_conversation(format='txt')
        self.assertIn('Hello', text_export)
        self.assertIn('joy', text_export)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""
    
    def test_end_to_end_workflow(self):
        """Test complete emotion detection -> response generation workflow."""
        from emotion_detector import EmotionDetector
        from response_generator import ResponseGenerator
        from safety_module import SafetyModule
        
        # Initialize components
        emotion_detector = EmotionDetector(num_labels=28)
        response_generator = ResponseGenerator(model_name="gpt2", max_length=50)
        safety_module = SafetyModule()
        
        # User message
        user_message = "I'm feeling anxious about my exams"
        
        # Detect emotion
        emotion_result = emotion_detector.predict_emotion(user_message)
        detected_emotion = emotion_result['emotion']
        
        # Assess safety
        risk_assessment = safety_module.assess_risk_level(
            text=user_message,
            emotion=detected_emotion
        )
        
        # Generate response
        response = response_generator.generate_response(
            user_message=user_message,
            detected_emotion=detected_emotion
        )
        
        # Validate response
        is_valid, reason = safety_module.validate_response(response)
        
        # Assertions
        self.assertIsNotNone(detected_emotion)
        self.assertIsNotNone(response)
        self.assertIsInstance(risk_assessment['overall_risk_score'], float)
        self.assertIsInstance(is_valid, bool)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestEmotionDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestSafetyModule))
    suite.addTests(loader.loadTestsFromTestCase(TestResponseGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestConversationManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed. Please review the output above.")
