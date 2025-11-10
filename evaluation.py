"""Evaluation system for RAG pipeline quality assessment."""
from typing import List, Dict, Optional
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
import pandas as pd
from logger import rag_logger


class RAGEvaluator:
    """Evaluate RAG system performance and quality."""
    
    def __init__(self, metrics_dir: str = "./metrics"):
        """Initialize evaluator.
        
        Args:
            metrics_dir: Directory to store metrics
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True)
        self.metrics_file = self.metrics_dir / "metrics.jsonl"
        
        # Runtime metrics
        self.query_count = 0
        self.cache_hits = 0
        self.total_response_time = 0.0
        self.error_count = 0
        
        rag_logger.info("RAG evaluator initialized")
    
    def evaluate_relevance(
        self,
        question: str,
        answer: str,
        sources: List[Dict],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate answer relevance and quality.
        
        Args:
            question: User question
            answer: System answer
            sources: Source documents used
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Answer length check
        metrics['answer_length'] = len(answer)
        metrics['has_answer'] = len(answer) > 0
        
        # Source usage
        metrics['num_sources'] = len(sources)
        metrics['sources_used'] = len(sources) > 0
        
        # Basic relevance (keyword overlap)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        if question_words:
            overlap = len(question_words & answer_words)
            metrics['keyword_overlap'] = overlap / len(question_words)
        else:
            metrics['keyword_overlap'] = 0.0
        
        # Answer completeness (heuristic)
        metrics['answer_completeness'] = min(len(answer) / 100, 1.0)
        
        # Source relevance
        if sources:
            source_texts = [s.get('content', '') for s in sources]
            source_words = set(' '.join(source_texts).lower().split())
            
            if source_words:
                source_overlap = len(answer_words & source_words)
                metrics['source_answer_overlap'] = min(source_overlap / len(answer_words), 1.0) if answer_words else 0.0
            else:
                metrics['source_answer_overlap'] = 0.0
        else:
            metrics['source_answer_overlap'] = 0.0
        
        # Ground truth comparison if provided
        if ground_truth:
            gt_words = set(ground_truth.lower().split())
            if gt_words:
                gt_overlap = len(answer_words & gt_words)
                metrics['ground_truth_overlap'] = gt_overlap / len(gt_words)
            else:
                metrics['ground_truth_overlap'] = 0.0
        
        # Overall quality score (weighted average)
        quality_score = (
            metrics['has_answer'] * 0.2 +
            metrics['sources_used'] * 0.2 +
            metrics['keyword_overlap'] * 0.2 +
            metrics['answer_completeness'] * 0.2 +
            metrics['source_answer_overlap'] * 0.2
        )
        metrics['quality_score'] = quality_score
        
        return metrics
    
    def log_query_metrics(
        self,
        question: str,
        answer: str,
        response_time: float,
        cache_hit: bool = False,
        num_sources: int = 0,
        error: Optional[str] = None
    ):
        """Log metrics for a query.
        
        Args:
            question: User question
            answer: System answer
            response_time: Response time in seconds
            cache_hit: Whether cache was used
            num_sources: Number of source documents
            error: Error message if any
        """
        self.query_count += 1
        if cache_hit:
            self.cache_hits += 1
        self.total_response_time += response_time
        if error:
            self.error_count += 1
        
        # Basic metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'question': question[:100],
            'answer_length': len(answer),
            'num_sources': num_sources,
            'response_time': response_time,
            'cache_hit': cache_hit,
            'error': error
        }
        
        # Save to file
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
        
        rag_logger.debug(f"Logged metrics: response_time={response_time:.2f}s, sources={num_sources}")
    
    def get_performance_summary(self, last_n_hours: int = 24) -> Dict:
        """Get performance summary.
        
        Args:
            last_n_hours: Number of hours to analyze
            
        Returns:
            Performance summary dict
        """
        summary = {
            'total_queries': self.query_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / self.query_count if self.query_count > 0 else 0,
            'avg_response_time': self.total_response_time / self.query_count if self.query_count > 0 else 0,
            'error_count': self.error_count,
            'error_rate': self.error_count / self.query_count if self.query_count > 0 else 0
        }
        
        # Load metrics from file
        if self.metrics_file.exists():
            try:
                metrics = []
                cutoff_time = datetime.now() - timedelta(hours=last_n_hours)
                
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            metric = json.loads(line)
                            timestamp = datetime.fromisoformat(metric['timestamp'])
                            if timestamp >= cutoff_time:
                                metrics.append(metric)
                        except:
                            continue
                
                if metrics:
                    df = pd.DataFrame(metrics)
                    
                    summary.update({
                        'avg_quality_score': df['quality_score'].mean(),
                        'avg_answer_length': df['answer_length'].mean(),
                        'avg_sources_used': df['num_sources'].mean(),
                        'recent_queries': len(metrics)
                    })
            except Exception as e:
                rag_logger.warning(f"Failed to load metrics: {e}")
        
        return summary
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate evaluation report.
        
        Args:
            output_file: Optional file to save report
            
        Returns:
            Report text
        """
        summary = self.get_performance_summary()
        
        report = f"""
╔══════════════════════════════════════════╗
║     RAG System Evaluation Report         ║
╚══════════════════════════════════════════╝

📊 Performance Metrics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Queries:        {summary['total_queries']}
Cache Hits:           {summary['cache_hits']} ({summary['cache_hit_rate']*100:.1f}%)
Avg Response Time:    {summary['avg_response_time']:.2f}s
Errors:               {summary['error_count']} ({summary['error_rate']*100:.1f}%)

📈 Quality Metrics (Last 24h)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recent Queries:       {summary.get('recent_queries', 0)}
Avg Quality Score:    {summary.get('avg_quality_score', 0):.2f}/1.0
Avg Answer Length:    {summary.get('avg_answer_length', 0):.0f} chars
Avg Sources Used:     {summary.get('avg_sources_used', 0):.1f}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            rag_logger.info(f"Report saved to: {output_file}")
        
        return report
    
    def reset_metrics(self):
        """Reset runtime metrics."""
        self.query_count = 0
        self.cache_hits = 0
        self.total_response_time = 0.0
        self.error_count = 0
        rag_logger.info("Reset runtime metrics")


if __name__ == "__main__":
    # Test evaluator
    evaluator = RAGEvaluator()
    
    # Test evaluation
    metrics = evaluator.evaluate_relevance(
        question="Galaxy S22의 배터리 용량은?",
        answer="Galaxy S22의 배터리 용량은 3,700mAh입니다.",
        sources=[{"content": "Galaxy S22 배터리: 3700mAh"}]
    )
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Test logging
    evaluator.log_query_metrics(
        question="Test question",
        answer="Test answer",
        sources=[{"content": "test"}],
        response_time=1.5,
        cache_hit=False
    )
    
    # Generate report
    report = evaluator.generate_report()
    print(report)
