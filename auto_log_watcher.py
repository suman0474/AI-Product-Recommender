"""
Auto Log Watcher - Runs as a BACKGROUND PROCESS
================================================
Automatically captures terminal/log data and stores it in TL1.md - TL5.md+

HOW TO USE:
-----------
1. Start this watcher in background:
   START /B python auto_log_watcher.py

2. Start your main app with output redirection:
   python main.py 2>&1 >> app_output.log

The watcher will automatically capture all output and store it in TL files.

ALTERNATIVE - All in one command:
   python auto_log_watcher.py --wrap "python main.py"
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
import signal

class AutoLogWatcher:
    """Automatically watches and stores log data."""
    
    MAX_LINES_PER_FILE = 5000
    FILE_PREFIX = "TL"
    FILE_EXTENSION = ".md"
    LOG_FILE = "app_output.log"
    
    def __init__(self):
        # Go to parent directory for TL files
        self.script_dir = Path(__file__).parent.resolve()
        if self.script_dir.name == 'backend':
            self.base_dir = self.script_dir.parent
        else:
            self.base_dir = self.script_dir
        
        self.log_file_path = self.script_dir / self.LOG_FILE
        self.current_file_index = 1
        self.running = True
        self.last_position = 0
        
        # Initialize TL files
        self._find_active_file()
        
        print(f"📁 Log storage: {self.base_dir}")
        print(f"📄 Watching: {self.log_file_path}")
        print(f"📝 Writing to: TL{self.current_file_index}.md")
        print("=" * 50)
    
    def _get_file_path(self, index: int) -> Path:
        return self.base_dir / f"{self.FILE_PREFIX}{index}{self.FILE_EXTENSION}"
    
    def _create_file(self, file_path: Path, file_num: int):
        header = f"""# Terminal Log {file_num}

> Auto-generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> Max lines: {self.MAX_LINES_PER_FILE}

---

"""
        file_path.write_text(header, encoding='utf-8')
        print(f"✨ Created: {file_path.name}")
    
    def _get_line_count(self, file_path: Path) -> int:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def _find_active_file(self):
        """Find or create the active file with space."""
        while True:
            file_path = self._get_file_path(self.current_file_index)
            if not file_path.exists():
                self._create_file(file_path, self.current_file_index)
                return
            if self._get_line_count(file_path) >= self.MAX_LINES_PER_FILE:
                self.current_file_index += 1
                continue
            return
    
    def _append_lines(self, lines: list):
        """Append lines to current TL file, rolling over if needed."""
        if not lines:
            return
        
        for line in lines:
            file_path = self._get_file_path(self.current_file_index)
            
            # Check if need to roll over
            if self._get_line_count(file_path) >= self.MAX_LINES_PER_FILE:
                self.current_file_index += 1
                file_path = self._get_file_path(self.current_file_index)
                self._create_file(file_path, self.current_file_index)
                print(f"📄 Rolled over to: {file_path.name}")
            
            # Append line
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
    
    def watch_log_file(self, interval: float = 1.0):
        """Watch the log file and store new content continuously."""
        print(f"👀 Watching for new content every {interval}s... (Ctrl+C to stop)")
        
        # Create log file if it doesn't exist
        if not self.log_file_path.exists():
            self.log_file_path.write_text("", encoding='utf-8')
            print(f"📄 Created empty log file: {self.log_file_path.name}")
        
        while self.running:
            try:
                with open(self.log_file_path, 'r', encoding='utf-8', errors='replace') as f:
                    f.seek(self.last_position)
                    new_content = f.read()
                    self.last_position = f.tell()
                    
                    if new_content.strip():
                        lines = new_content.strip().split('\n')
                        self._append_lines(lines)
                        print(f"📝 Stored {len(lines)} lines → TL{self.current_file_index}.md")
                
                time.sleep(interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                time.sleep(interval)
        
        print("\n👋 Watcher stopped.")
    
    def wrap_and_run(self, command: str):
        """Run a command and capture ALL its output to TL files automatically."""
        print(f"🚀 Running: {command}")
        print("📝 All output will be stored automatically...")
        print("=" * 50)
        
        # Add timestamp header
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._append_lines([f"\n## [{timestamp}] - Command: {command}\n"])
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )
        
        try:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    line = line.rstrip()
                    print(line)  # Show in terminal
                    self._append_lines([line])  # Store in TL file
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
            process.terminate()
        
        print("\n✅ Process finished. All output stored.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto Log Watcher")
    parser.add_argument('--wrap', '-w', type=str, help='Wrap and run a command, capturing all output')
    parser.add_argument('--watch', action='store_true', help='Watch app_output.log for new content')
    parser.add_argument('--interval', '-i', type=float, default=1.0, help='Watch interval in seconds')
    
    args = parser.parse_args()
    watcher = AutoLogWatcher()
    
    if args.wrap:
        watcher.wrap_and_run(args.wrap)
    elif args.watch:
        watcher.watch_log_file(args.interval)
    else:
        # Default: wrap mode instruction
        print("""
╔══════════════════════════════════════════════════════════════╗
║           AUTO LOG WATCHER - Usage Instructions              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  OPTION 1: Wrap your command (RECOMMENDED)                   ║
║  ─────────────────────────────────────────                   ║
║  python auto_log_watcher.py --wrap "python main.py"          ║
║                                                              ║
║  This runs your app AND stores all output automatically.     ║
║                                                              ║
║  OPTION 2: Watch a log file                                  ║
║  ────────────────────────────                                ║
║  Terminal 1: python main.py >> app_output.log 2>&1           ║
║  Terminal 2: python auto_log_watcher.py --watch              ║
║                                                              ║
║  OPTION 3: Background watching                               ║
║  ────────────────────────────                                ║
║  START /B python auto_log_watcher.py --watch                 ║
║  python main.py >> app_output.log 2>&1                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    main()
