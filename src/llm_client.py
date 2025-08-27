import logging
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import torch
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the LLM model"""
        try:
            model_name = settings.llm_model_name
            
            # Determine if it's a seq2seq or causal LM model
            if "t5" in model_name.lower() or "flan" in model_name.lower():
                # Seq2Seq model (T5, FLAN-T5)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                task = "text2text-generation"
            else:
                # Causal LM model (GPT, DialoGPT, etc.)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                task = "text-generation"
            
            # Create pipeline
            self.pipeline = pipeline(
                task,
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info(f"Loaded LLM model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading LLM model: {e}")
            # Fallback to a smaller model
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model if the main model fails"""
        try:
            # Use a smaller, reliable model as fallback
            fallback_model = "google/flan-t5-small"
            
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model)
            
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            
            logger.info(f"Loaded fallback LLM model: {fallback_model}")
            
        except Exception as e:
            logger.error(f"Error loading fallback model: {e}")
            raise
    
    def generate_answer(self, question: str, context: List[Dict[str, Any]], 
                       max_length: int = 512) -> str:
        """Generate answer based on question and retrieved context"""
        if not self.pipeline:
            return "I apologize, but the language model is not available."
        
        try:
            # Build prompt with context
            prompt = self._build_prompt(question, context)
            
            # Generate response
            if "text2text-generation" in str(type(self.pipeline)):
                # For T5-style models
                response = self.pipeline(
                    prompt,
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                answer = response[0]['generated_text'].strip()
            else:
                # For GPT-style models
                response = self.pipeline(
                    prompt,
                    max_new_tokens=200,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                # Extract only the new generated text
                full_text = response[0]['generated_text']
                answer = full_text[len(prompt):].strip()
            
            # Post-process the answer
            answer = self._post_process_answer(answer, question, context)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer."
    
    def _build_prompt(self, question: str, context: List[Dict[str, Any]]) -> str:
        """Build prompt for the LLM"""
        if not context:
            return f"Question: {question}\nAnswer: I don't have enough information to answer this question."
        
        # Build context section
        context_text = ""
        for i, doc in enumerate(context[:3], 1):  # Use top 3 documents
            text = doc.get('text', '')[:500]  # Limit context length
            filename = doc.get('metadata', {}).get('filename', 'Unknown')
            context_text += f"Document {i} ({filename}):\n{text}\n\n"
        
        # Create prompt based on model type
        if "t5" in settings.llm_model_name.lower() or "flan" in settings.llm_model_name.lower():
            # T5-style prompt
            prompt = f"""Answer the following question based only on the provided context. If the context doesn't contain enough information to answer the question, say "I don't know."

Context:
{context_text}

Question: {question}

Answer:"""
        else:
            # GPT-style prompt
            prompt = f"""You are a helpful assistant. Answer the following question based only on the provided context. If the context doesn't contain enough information to answer the question, say "I don't know."

Context:
{context_text}

Question: {question}
Answer:"""
        
        return prompt
    
    def _post_process_answer(self, answer: str, question: str, context: List[Dict[str, Any]]) -> str:
        """Post-process the generated answer"""
        # Clean up the answer
        answer = answer.strip()
        
        # Remove any repetitive patterns
        lines = answer.split('\n')
        unique_lines = []
        seen = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen:
                unique_lines.append(line)
                seen.add(line)
        
        answer = '\n'.join(unique_lines)
        
        # Truncate if too long
        if len(answer) > 1000:
            sentences = answer.split('. ')
            truncated = '. '.join(sentences[:3])
            if not truncated.endswith('.'):
                truncated += '.'
            answer = truncated
        
        # Check if answer is grounded in context
        if not self._is_answer_grounded(answer, context):
            return "I don't have enough reliable information in the provided documents to answer this question accurately."
        
        return answer
    
    def _is_answer_grounded(self, answer: str, context: List[Dict[str, Any]]) -> bool:
        """Check if the answer is grounded in the provided context"""
        if not context or not answer:
            return False
        
        if "don't know" in answer.lower() or "don't have" in answer.lower():
            return True  # These are acceptable responses
        
        # Simple keyword overlap check
        answer_words = set(answer.lower().split())
        context_words = set()
        
        for doc in context:
            text = doc.get('text', '').lower()
            context_words.update(text.split())
        
        # Check if there's reasonable overlap
        overlap = len(answer_words.intersection(context_words))
        overlap_ratio = overlap / len(answer_words) if answer_words else 0
        
        return overlap_ratio > 0.3  # At least 30% word overlap
    
    def is_model_loaded(self) -> bool:
        """Check if the model is properly loaded"""
        return self.pipeline is not None