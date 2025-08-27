import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GuardrailsManager:
    def __init__(self):
        self.unsafe_patterns = [
            # Harmful content patterns
            r'\b(?:kill|murder|suicide|harm|hurt|violence|weapon|bomb|terrorist)\b',
            r'\b(?:hate|racist|sexist|discrimination|abuse)\b',
            r'\b(?:illegal|fraud|scam|steal|hack|piracy)\b',
            r'\b(?:drug|cocaine|heroin|meth|marijuana)\b',
            # Personal information patterns
            r'\b(?:ssn|social security|credit card|password|pin)\b',
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card pattern
        ]
        
        self.off_topic_patterns = [
            r'\b(?:weather|sports|cooking|recipe|movie|music|game)\b',
            r'\b(?:personal|family|relationship|dating|love)\b',
            r'\b(?:politics|election|vote|campaign|politician)\b',
        ]
        
        self.grounding_keywords = [
            'according to', 'based on', 'the document states', 'from the text',
            'as mentioned', 'the source indicates', 'referenced in'
        ]
    
    def check_query_safety(self, query: str) -> Dict[str, Any]:
        """Check if a query is safe and appropriate"""
        query_lower = query.lower()
        
        result = {
            'is_safe': True,
            'is_on_topic': True,
            'reason': '',
            'should_refuse': False
        }
        
        # Check for unsafe content
        for pattern in self.unsafe_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                result['is_safe'] = False
                result['should_refuse'] = True
                result['reason'] = 'Query contains potentially harmful content'
                return result
        
        # Check for off-topic content (less strict)
        off_topic_count = 0
        for pattern in self.off_topic_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                off_topic_count += 1
        
        if off_topic_count > 1:  # Multiple indicators of off-topic
            result['is_on_topic'] = False
            result['reason'] = 'Query appears to be off-topic for document search'
        
        return result
    
    def validate_answer_grounding(self, answer: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if the answer is properly grounded in the context"""
        result = {
            'is_grounded': True,
            'confidence': 1.0,
            'reason': '',
            'should_modify': False
        }
        
        if not context:
            result['is_grounded'] = False
            result['confidence'] = 0.0
            result['reason'] = 'No context provided'
            result['should_modify'] = True
            return result
        
        answer_lower = answer.lower()
        
        # Check if answer explicitly says "I don't know" - this is acceptable
        if any(phrase in answer_lower for phrase in ["don't know", "don't have", "insufficient", "not enough"]):
            result['confidence'] = 1.0
            result['reason'] = 'Appropriate response for insufficient information'
            return result
        
        # Check for grounding indicators
        has_grounding_indicators = any(
            keyword in answer_lower for keyword in self.grounding_keywords
        )
        
        # Calculate content overlap
        answer_words = set(re.findall(r'\b\w+\b', answer_lower))
        context_words = set()
        
        for doc in context:
            text = doc.get('text', '').lower()
            context_words.update(re.findall(r'\b\w+\b', text))
        
        if not answer_words:
            result['is_grounded'] = False
            result['confidence'] = 0.0
            result['reason'] = 'Empty or invalid answer'
            result['should_modify'] = True
            return result
        
        # Calculate overlap ratio
        overlap = len(answer_words.intersection(context_words))
        overlap_ratio = overlap / len(answer_words)
        
        # Determine confidence based on multiple factors
        confidence_factors = []
        
        # Factor 1: Word overlap
        if overlap_ratio >= 0.5:
            confidence_factors.append(0.4)
        elif overlap_ratio >= 0.3:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.0)
        
        # Factor 2: Grounding indicators
        if has_grounding_indicators:
            confidence_factors.append(0.3)
        else:
            confidence_factors.append(0.0)
        
        # Factor 3: Answer length appropriateness
        if 10 <= len(answer.split()) <= 100:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
        
        # Factor 4: Context relevance
        context_length = sum(len(doc.get('text', '').split()) for doc in context)
        if context_length > 50:
            confidence_factors.append(0.1)
        else:
            confidence_factors.append(0.0)
        
        result['confidence'] = sum(confidence_factors)
        
        # Determine if answer is grounded
        if result['confidence'] < 0.3:
            result['is_grounded'] = False
            result['should_modify'] = True
            result['reason'] = f'Low confidence score: {result["confidence"]:.2f}. Insufficient grounding in context.'
        elif result['confidence'] < 0.5:
            result['reason'] = f'Medium confidence: {result["confidence"]:.2f}. Answer may be partially grounded.'
        else:
            result['reason'] = f'High confidence: {result["confidence"]:.2f}. Answer appears well-grounded.'
        
        return result
    
    def apply_guardrails(self, query: str, answer: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply all guardrails and return final decision"""
        # Check query safety
        query_check = self.check_query_safety(query)
        
        if query_check['should_refuse']:
            return {
                'final_answer': "I cannot provide information on that topic. Please ask questions related to the available documents.",
                'is_safe': False,
                'is_grounded': False,
                'reason': query_check['reason'],
                'query_check': query_check,
                'grounding_check': None
            }
        
        # Check answer grounding
        grounding_check = self.validate_answer_grounding(answer, context)
        
        final_answer = answer
        
        if grounding_check['should_modify']:
            if not context:
                final_answer = "I don't have any relevant documents to answer this question."
            else:
                final_answer = "I don't have enough reliable information in the provided documents to answer this question accurately."
        
        return {
            'final_answer': final_answer,
            'is_safe': query_check['is_safe'],
            'is_grounded': grounding_check['is_grounded'],
            'confidence': grounding_check.get('confidence', 0.0),
            'reason': grounding_check.get('reason', ''),
            'query_check': query_check,
            'grounding_check': grounding_check
        }
    
    def format_answer_with_citations(self, answer: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format answer with proper citations"""
        if not context:
            return {
                'answer': answer,
                'citations': [],
                'sources': []
            }
        
        citations = []
        sources = []
        
        for i, doc in enumerate(context, 1):
            metadata = doc.get('metadata', {})
            filename = metadata.get('filename', 'Unknown Document')
            page_number = metadata.get('page_number', 'Unknown')
            drive_url = metadata.get('drive_url', '#')
            text_snippet = doc.get('text', '')[:200] + '...' if len(doc.get('text', '')) > 200 else doc.get('text', '')
            
            citation = {
                'id': i,
                'filename': filename,
                'page_number': page_number,
                'url': drive_url,
                'snippet': text_snippet,
                'relevance_score': doc.get('score', 0.0)
            }
            
            citations.append(citation)
            sources.append(f"[{i}] {filename} (Page {page_number})")
        
        # Add citation references to answer if not present
        if citations and not any(f"[{i}]" in answer for i in range(1, len(citations) + 1)):
            answer += f"\n\nSources: {', '.join([f'[{c[\"id\"]}]' for c in citations])}"
        
        return {
            'answer': answer,
            'citations': citations,
            'sources': sources
        }
    
    def log_interaction(self, query: str, answer: str, safety_result: Dict[str, Any]):
        """Log interaction for monitoring"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'query': query[:100] + '...' if len(query) > 100 else query,
            'answer_length': len(answer),
            'is_safe': safety_result['is_safe'],
            'is_grounded': safety_result['is_grounded'],
            'confidence': safety_result.get('confidence', 0.0),
            'reason': safety_result.get('reason', '')
        }
        
        logger.info(f"Interaction logged: {log_entry}")