"""
Logging utilities
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json

def setup_logger(name: str, log_dir: str = "./logs", level=logging.INFO):
    """
    Setup logger với file và console handlers
    
    Args:
        name: Logger name
        log_dir: Directory to save logs
        level: Logging level
        
    Returns:
        logger instance
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler (detailed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_path / f"processing_{timestamp}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (simple)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class StatisticsTracker:
    """Track và save statistics"""
    
    def __init__(self, output_dir: str = "./logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'timestamp': datetime.now().isoformat(),
            'approaches': {}
        }
    
    def add_approach_stats(self, approach_name: str, stats_dict: dict):
        """Add statistics cho một approach"""
        self.stats['approaches'][approach_name] = stats_dict
    
    def save(self):
        """Save statistics to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"statistics_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        return output_file
    
    def print_summary(self):
        """Print summary của tất cả approaches"""
        print("\n" + "="*70)
        print("📊 STATISTICS SUMMARY")
        print("="*70)
        
        for approach_name, stats in self.stats['approaches'].items():
            print(f"\n{approach_name.upper()}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")