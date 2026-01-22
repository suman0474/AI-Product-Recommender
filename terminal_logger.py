"""
Terminal Logger Script
======================
Captures terminal data and stores each word into TL*.md files.
- Max 5000 lines per file.
- Automatically creates new files (TL6, TL7, etc.) when previous ones are full.
- Supports real-time concurrent logging via the --run command.

Usage:
    1. Real-time logging (Recommended):
       python terminal_logger.py --run "python script.py"
       
    2. Pipe output: 
       command | python terminal_logger.py
       
    3. Interactive: 
       python terminal_logger.py --interactive
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import time

class TerminalLogger:
    """Stores terminal data word by word into multiple markdown files."""
    
    # Configuration
    MAX_LINES_PER_FILE = 5000
    FILE_PREFIX = "TL"
    FILE_EXTENSION = ".md"
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the logger."""
        # Default to parent directory if in backend, or current directory
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            current_dir = Path(__file__).parent.resolve()
            if current_dir.name == 'backend':
                self.base_dir = current_dir.parent
            else:
                self.base_dir = current_dir
        
        print(f"📁 Log storage directory: {self.base_dir}")
        self.current_file_index = 1
        
        # Ensure at least TL1 exists to start
        self._ensure_file_ready(1)
        
        # Find the current active file (the first one that isn't full)
        self._find_active_file()
    
    def _find_active_file(self):
        """Find the first file that has space, creating new ones if needed."""
        while True:
            file_path = self._get_file_path(self.current_file_index)
            
            # If file doesn't exist, create it and use it
            if not file_path.exists():
                self._create_file(file_path, self.current_file_index)
                return

            # If file exists but is full, move to next
            if self._get_line_count(file_path) >= self.MAX_LINES_PER_FILE:
                self.current_file_index += 1
                continue
            
            # File exists and has space
            return
            
    def _get_file_path(self, index: int) -> Path:
        return self.base_dir / f"{self.FILE_PREFIX}{index}{self.FILE_EXTENSION}"
    
    def _ensure_file_ready(self, index: int):
        """Ensure a specific file number exists."""
        file_path = self._get_file_path(index)
        if not file_path.exists():
            self._create_file(file_path, index)
    
    def _create_file(self, file_path: Path, file_num: int):
        """Create a new log file with header."""
        header = f"""# Terminal Log {file_num}

> Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> Max lines: {self.MAX_LINES_PER_FILE}

---

"""
        file_path.write_text(header, encoding='utf-8')
        print(f"✨ Created new log file: {file_path.name}")
    
    def _get_line_count(self, file_path: Path) -> int:
        """Get the current line count of a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return 0
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract all words from text, preserving meaningful tokens."""
        words = text.split()
        return [word for word in words if word.strip()]
    
    def _append_to_file(self, file_path: Path, content: str):
        """Append content to a file."""
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(content)
    
    def log_words(self, text: str, words_per_line: int = 10) -> dict:
        """Store each word from the terminal data into the log files."""
        words = self._extract_words(text)
        
        if not words:
            return {"status": "empty", "words_logged": 0}
        
        stats = {
            "total_words": len(words),
            "words_logged": 0,
            "files_used": set(),
            "lines_written": 0
        }
        
        word_index = 0
        
        while word_index < len(words):
            # 1. Get current active file
            file_path = self._get_file_path(self.current_file_index)
            
            # 2. Check existence/creation
            if not file_path.exists():
                self._create_file(file_path, self.current_file_index)
            
            # 3. Check capacity
            current_lines = self._get_line_count(file_path)
            available_lines = self.MAX_LINES_PER_FILE - current_lines
            
            if available_lines <= 0:
                # Current file is full, move to next
                self.current_file_index += 1
                continue
            
            # 4. Write chunk
            max_words_capacity = available_lines * words_per_line
            remaining_words = len(words) - word_index
            words_to_write_count = min(max_words_capacity, remaining_words)
            
            chunk_end_index = word_index + words_to_write_count
            lines_content = []
            
            for i in range(word_index, chunk_end_index, words_per_line):
                line_words = words[i:min(i + words_per_line, chunk_end_index)]
                lines_content.append(" ".join(line_words))
                stats["lines_written"] += 1
            
            content = "\n".join(lines_content) + "\n"
            self._append_to_file(file_path, content)
            
            stats["files_used"].add(file_path.name)
            stats["words_logged"] += words_to_write_count
            word_index += words_to_write_count
            
        stats["files_used"] = list(stats["files_used"])
        return stats

    def log_with_timestamp(self, text: str, source: str = "Terminal") -> dict:
        """Log words with a timestamp header."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header = f"\n\n## [{timestamp}] - {source}\n\n"
        
        # Write header to active file
        self._find_active_file() # Ensure we are pointing to a valid file
        file_path = self._get_file_path(self.current_file_index)
        
        # If current file is strictly full, force next for header?
        # Better: let log_words handle rollover, but header should go to a file with space.
        # Simple check:
        if self._get_line_count(file_path) >= self.MAX_LINES_PER_FILE:
            self.current_file_index += 1
            file_path = self._get_file_path(self.current_file_index)
            self._create_file(file_path, self.current_file_index)
            
        self._append_to_file(file_path, header)
        return self.log_words(text)
        
    def run_command(self, command: str):
        """Run a command and capture its output to files in real-time (concurrently)."""
        print(f"🚀 Running: {command}")
        print("📝 Logging output concurrently to TL files...")
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered for real-time processing
            encoding='utf-8',
            errors='replace'
        )
        
        # Log start
        self.log_with_timestamp("", f"Command Start: {command}")
        
        try:
            while True:
                # Read line by line for real-time logging
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                
                if line:
                    # Echo to console so user sees it live
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    # Store in file concurrently
                    self.log_words(line)
        except KeyboardInterrupt:
            print("\n🛑 Execution interrupted by user.")
            process.kill()
            
        print("\n✅ Command finished.")

def main():
    parser = argparse.ArgumentParser(description="Terminal Data Logger (Auto-expanding)")
    parser.add_argument('--input', '-i', type=str, help='Input file to read from')
    parser.add_argument('--interactive', action='store_true', help='Interactive input mode')
    parser.add_argument('--status', '-s', action='store_true', help='Show status of log files')
    parser.add_argument('--run', '-r', type=str, help='Run a command and log output concurrently')
    parser.add_argument('--words-per-line', '-w', type=int, default=10, help='Words per line')
    
    args = parser.parse_args()
    
    logger = TerminalLogger()
    
    if args.status:
        # Show status of all existing TL files
        i = 1
        total_lines = 0
        while True:
            path = logger._get_file_path(i)
            if not path.exists():
                break
            lines = logger._get_line_count(path)
            total_lines += lines
            bar = "█" * int((lines / logger.MAX_LINES_PER_FILE) * 10)
            print(f"{path.name:<10} {lines:>4}/{logger.MAX_LINES_PER_FILE} lines {bar}")
            i += 1
        print(f"Total lines: {total_lines}")
        return

    if args.run:
        # This checks the box for "generated data should store... concurrently"
        logger.run_command(args.run)
        return

    if args.input:
        text = Path(args.input).read_text(encoding='utf-8', errors='replace')
        logger.log_with_timestamp(text, f"File: {args.input}")
        return

    if args.interactive:
        print("Interactive Mode. Type and press Enter.")
        try:
            while True:
                line = input()
                logger.log_words(line)
        except KeyboardInterrupt:
            return

    # Pipe mode
    if not sys.stdin.isatty():
        # Read chunk by chunk? Or all at once?
        # For pipes, we might want real-time too.
        # But sys.stdin.read() blocks until EOF.
        # Use readline loop for pipes to support streaming logs via pipe
        header_logged = False
        try:
            for line in sys.stdin:
                if not header_logged:
                     logger.log_with_timestamp("", "Piped Input")
                     header_logged = True
                logger.log_words(line)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
