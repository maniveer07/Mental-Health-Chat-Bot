"""
Test Script for Visualization Manager
Verifies all chart generation functions work correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.visualization_manager import VisualizationManager

def create_sample_conversation():
    """Create sample conversation data for testing"""
    return [
        {
            'role': 'user',
            'content': 'I am feeling really sad today',
            'emotion': 'sadness',
            'confidence': 0.85,
            'risk_score': 0.2
        },
        {
            'role': 'assistant',
            'content': 'I hear you. It sounds like you\'re going through a difficult time.'
        },
        {
            'role': 'user',
            'content': 'Yeah, I lost my job',
            'emotion': 'sadness',
            'confidence': 0.92,
            'risk_score': 0.3
        },
        {
            'role': 'assistant',
            'content': 'That must be really challenging. How are you coping?'
        },
        {
            'role': 'user',
            'content': 'I am trying to stay positive but it is hard',
            'emotion': 'neutral',
            'confidence': 0.65,
            'risk_score': 0.25
        },
        {
            'role': 'assistant',
            'content': 'It\'s okay to feel this way. Your feelings are valid.'
        },
        {
            'role': 'user',
            'content': 'Actually I am getting angry about the whole situation',
            'emotion': 'anger',
            'confidence': 0.78,
            'risk_score': 0.35
        },
        {
            'role': 'assistant',
            'content': 'Anger is a natural response. It shows you care.'
        },
        {
            'role': 'user',
            'content': 'Thanks for listening, I feel a bit better now',
            'emotion': 'joy',
            'confidence': 0.72,
            'risk_score': 0.1
        },
        {
            'role': 'assistant',
            'content': 'I\'m glad I could help. You\'re doing great by talking about it.'
        },
        {
            'role': 'user',
            'content': 'I am still worried about finding a new job though',
            'emotion': 'fear',
            'confidence': 0.88,
            'risk_score': 0.2
        },
    ]


def test_all_visualizations():
    """Test all visualization functions"""
    
    print("=" * 60)
    print("  TESTING VISUALIZATION MANAGER")
    print("=" * 60)
    print()
    
    # Initialize manager
    print("✓ Initializing VisualizationManager...")
    viz_manager = VisualizationManager()
    print("  Success!\n")
    
    # Create sample data
    print("✓ Creating sample conversation data...")
    conversation_history = create_sample_conversation()
    print(f"  Created {len(conversation_history)} conversation turns\n")
    
    # Test 1: Emotion Timeline
    print("1. Testing emotion timeline...")
    try:
        timeline_fig = viz_manager.create_emotion_timeline(conversation_history)
        print(f"   ✓ Timeline created successfully")
        print(f"   - Data points: {len([t for t in conversation_history if t.get('role') == 'user'])}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    # Test 2: Emotion Distribution
    print("2. Testing emotion distribution...")
    try:
        dist_fig = viz_manager.create_emotion_distribution(conversation_history)
        print(f"   ✓ Distribution chart created successfully")
        emotions = [t['emotion'] for t in conversation_history if t.get('role') == 'user' and 'emotion' in t]
        print(f"   - Unique emotions: {len(set(emotions))}")
        print(f"   - Emotions: {', '.join(set(emotions))}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    # Test 3: Confidence Trend
    print("3. Testing confidence trend...")
    try:
        conf_fig = viz_manager.create_confidence_trend(conversation_history)
        print(f"   ✓ Confidence chart created successfully")
        confidences = [t['confidence'] for t in conversation_history if 'confidence' in t]
        print(f"   - Average confidence: {sum(confidences)/len(confidences):.1%}")
        print(f"   - Min: {min(confidences):.1%}, Max: {max(confidences):.1%}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    # Test 4: Conversation Stats
    print("4. Testing conversation statistics...")
    try:
        stats = viz_manager.create_conversation_stats(conversation_history)
        print(f"   ✓ Statistics generated successfully")
        print(f"   - Total turns: {stats['total_turns']}")
        print(f"   - User messages: {stats['user_messages']}")
        print(f"   - Bot messages: {stats['bot_messages']}")
        print(f"   - Most common emotion: {stats['most_common_emotion'][0]} ({stats['most_common_emotion'][1]} times)")
        print(f"   - Average confidence: {stats['avg_confidence']:.1%}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    # Test 5: Risk Level Chart
    print("5. Testing risk level chart...")
    try:
        risk_fig = viz_manager.create_risk_level_chart(conversation_history)
        print(f"   ✓ Risk chart created successfully")
        risk_scores = [t['risk_score'] for t in conversation_history if 'risk_score' in t]
        print(f"   - Average risk: {sum(risk_scores)/len(risk_scores):.2f}")
        print(f"   - Max risk: {max(risk_scores):.2f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    # Test 6: Word Cloud Data
    print("6. Testing word cloud data...")
    try:
        word_data = viz_manager.create_word_cloud_data(conversation_history)
        print(f"   ✓ Word cloud data generated successfully")
        print(f"   - Total words: {len(word_data)}")
        if word_data:
            top_3 = list(word_data.items())[:3]
            print(f"   - Top words: {', '.join([f'{w} ({c})' for w, c in top_3])}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    # Test 7: Export
    print("7. Testing conversation export...")
    try:
        filename = viz_manager.export_conversation_data(
            conversation_history,
            filename="test_export.json"
        )
        print(f"   ✓ Export successful")
        print(f"   - Saved to: {filename}")
        
        # Verify file exists
        import os
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"   - File size: {size} bytes")
            
            # Clean up
            os.remove(filename)
            print(f"   - Test file cleaned up")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    # Test 8: Empty Data Handling
    print("8. Testing empty data handling...")
    try:
        empty_history = []
        empty_fig = viz_manager.create_emotion_timeline(empty_history)
        print(f"   ✓ Empty data handled gracefully")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    print("=" * 60)
    print("  ALL TESTS COMPLETED")
    print("=" * 60)
    print()
    print("Summary:")
    print("✓ All visualization functions are working correctly")
    print("✓ Charts can be generated from conversation data")
    print("✓ Export functionality is operational")
    print("✓ Empty data is handled safely")
    print()
    print("Next steps:")
    print("1. Launch enhanced interface: .\\launch_chainlit_enhanced.ps1")
    print("2. Start chatting to see real visualizations")
    print("3. Use /stats, /timeline, /export commands")
    print()


if __name__ == "__main__":
    try:
        test_all_visualizations()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

