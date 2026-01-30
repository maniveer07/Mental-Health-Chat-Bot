"""
Visualization Manager for Mental Health Chatbot
Generates charts and visual analytics for conversation data
"""

import json
from typing import List, Dict, Any
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter


class VisualizationManager:
    """Manages all visualizations for the chatbot"""
    
    def __init__(self):
        self.emotion_colors = {
            'anger': '#FF6B6B',
            'fear': '#9B59B6',
            'joy': '#FFD93D',
            'neutral': '#95A5A6',
            'sadness': '#3498DB',
            'surprise': '#FF8C42'
        }
    
    def create_emotion_timeline(self, conversation_history: List[Dict]) -> go.Figure:
        """
        Create a line graph showing emotion changes over the conversation
        
        Args:
            conversation_history: List of conversation turns with emotions
            
        Returns:
            Plotly figure object
        """
        timestamps = []
        emotions = []
        confidences = []
        
        for i, turn in enumerate(conversation_history):
            if turn.get('role') == 'user' and 'emotion' in turn:
                timestamps.append(i + 1)  # Turn number
                emotions.append(turn['emotion'])
                confidences.append(turn.get('confidence', 0))
        
        if not timestamps:
            return self._create_empty_chart("No emotion data yet")
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot with colors
        emotion_numeric = [list(self.emotion_colors.keys()).index(e) for e in emotions]
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=emotion_numeric,
            mode='lines+markers',
            marker=dict(
                size=10,
                color=[self.emotion_colors[e] for e in emotions],
                line=dict(width=2, color='white')
            ),
            line=dict(width=2, color='gray'),
            text=emotions,
            hovertemplate='<b>Turn %{x}</b><br>Emotion: %{text}<br>Confidence: %{customdata:.1%}<extra></extra>',
            customdata=confidences
        ))
        
        # Update layout
        fig.update_layout(
            title="Emotion Timeline",
            xaxis_title="Conversation Turn",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(self.emotion_colors))),
                ticktext=list(self.emotion_colors.keys())
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_emotion_distribution(self, conversation_history: List[Dict]) -> go.Figure:
        """
        Create a pie chart showing overall emotion distribution
        
        Args:
            conversation_history: List of conversation turns with emotions
            
        Returns:
            Plotly figure object
        """
        emotions = [turn['emotion'] for turn in conversation_history 
                   if turn.get('role') == 'user' and 'emotion' in turn]
        
        if not emotions:
            return self._create_empty_chart("No emotion data yet")
        
        emotion_counts = Counter(emotions)
        
        fig = go.Figure(data=[go.Pie(
            labels=list(emotion_counts.keys()),
            values=list(emotion_counts.values()),
            marker=dict(colors=[self.emotion_colors[e] for e in emotion_counts.keys()]),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Overall Emotion Distribution",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        return fig
    
    def create_confidence_trend(self, conversation_history: List[Dict]) -> go.Figure:
        """
        Create a line graph showing model confidence over time
        
        Args:
            conversation_history: List of conversation turns with confidence scores
            
        Returns:
            Plotly figure object
        """
        timestamps = []
        confidences = []
        emotions = []
        
        for i, turn in enumerate(conversation_history):
            if turn.get('role') == 'user' and 'confidence' in turn:
                timestamps.append(i + 1)
                confidences.append(turn['confidence'])
                emotions.append(turn.get('emotion', 'unknown'))
        
        if not timestamps:
            return self._create_empty_chart("No confidence data yet")
        
        # Calculate average
        avg_confidence = sum(confidences) / len(confidences)
        
        fig = go.Figure()
        
        # Add confidence line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=confidences,
            mode='lines+markers',
            name='Confidence',
            marker=dict(size=8, color='lightblue'),
            line=dict(width=3, color='blue'),
            hovertemplate='<b>Turn %{x}</b><br>Confidence: %{y:.1%}<br>Emotion: %{text}<extra></extra>',
            text=emotions
        ))
        
        # Add average line
        fig.add_trace(go.Scatter(
            x=[min(timestamps), max(timestamps)],
            y=[avg_confidence, avg_confidence],
            mode='lines',
            name=f'Average ({avg_confidence:.1%})',
            line=dict(dash='dash', color='red', width=2)
        ))
        
        fig.update_layout(
            title="Model Confidence Over Time",
            xaxis_title="Conversation Turn",
            yaxis_title="Confidence",
            yaxis=dict(tickformat='.0%'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_conversation_stats(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """
        Generate conversation statistics
        
        Args:
            conversation_history: List of conversation turns
            
        Returns:
            Dictionary with statistics
        """
        user_messages = [t for t in conversation_history if t.get('role') == 'user']
        bot_messages = [t for t in conversation_history if t.get('role') == 'assistant']
        
        emotions = [t['emotion'] for t in user_messages if 'emotion' in t]
        confidences = [t['confidence'] for t in user_messages if 'confidence' in t]
        
        stats = {
            'total_turns': len(user_messages),
            'total_messages': len(conversation_history),
            'user_messages': len(user_messages),
            'bot_messages': len(bot_messages),
            'emotions_detected': len(emotions),
            'most_common_emotion': Counter(emotions).most_common(1)[0] if emotions else ('N/A', 0),
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
        }
        
        return stats
    
    def create_risk_level_chart(self, conversation_history: List[Dict]) -> go.Figure:
        """
        Create a bar chart showing risk levels over time
        
        Args:
            conversation_history: List with risk assessments
            
        Returns:
            Plotly figure object
        """
        timestamps = []
        risk_scores = []
        
        for i, turn in enumerate(conversation_history):
            if turn.get('role') == 'user' and 'risk_score' in turn:
                timestamps.append(i + 1)
                risk_scores.append(turn['risk_score'])
        
        if not timestamps:
            return self._create_empty_chart("No risk data yet")
        
        # Color based on risk level
        colors = []
        for score in risk_scores:
            if score < 0.3:
                colors.append('green')
            elif score < 0.6:
                colors.append('yellow')
            else:
                colors.append('red')
        
        fig = go.Figure(data=[go.Bar(
            x=timestamps,
            y=risk_scores,
            marker=dict(color=colors),
            hovertemplate='<b>Turn %{x}</b><br>Risk Score: %{y:.2f}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Safety Risk Scores",
            xaxis_title="Conversation Turn",
            yaxis_title="Risk Score (0-1)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        return fig
    
    def create_word_cloud_data(self, conversation_history: List[Dict]) -> Dict[str, int]:
        """
        Generate word frequency data for word cloud
        
        Args:
            conversation_history: List of conversation turns
            
        Returns:
            Dictionary mapping words to frequencies
        """
        from collections import Counter
        import re
        
        # Get all user messages
        user_texts = [turn['content'] for turn in conversation_history 
                     if turn.get('role') == 'user']
        
        if not user_texts:
            return {}
        
        # Combine and clean text
        combined_text = ' '.join(user_texts).lower()
        
        # Remove common stop words
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'the', 'a', 'an', 
                     'and', 'or', 'but', 'is', 'am', 'are', 'was', 'were', 'be', 'been',
                     'to', 'of', 'in', 'for', 'on', 'with', 'as', 'by', 'at', 'from',
                     'that', 'this', 'it', 'its'}
        
        # Extract words
        words = re.findall(r'\b[a-z]{3,}\b', combined_text)
        words = [w for w in words if w not in stop_words]
        
        return dict(Counter(words).most_common(30))
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty placeholder chart"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        return fig
    
    def export_conversation_data(self, conversation_history: List[Dict], 
                                 filename: str = None) -> str:
        """
        Export conversation data to JSON file
        
        Args:
            conversation_history: List of conversation turns
            filename: Output filename (default: auto-generated)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_export_{timestamp}.json"
        
        data = {
            'export_date': datetime.now().isoformat(),
            'stats': self.create_conversation_stats(conversation_history),
            'conversation': conversation_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filename

