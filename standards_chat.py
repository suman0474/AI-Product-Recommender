#!/usr/bin/env python3
"""
Standards RAG System - Interactive CLI

Ask questions about industrial instrumentation standards and get
AI-generated answers with citations from the standards documents.

Usage:
    python standards_chat.py
"""

import sys
import os
import json
import uuid
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentic.standards_rag_workflow import run_standards_rag_workflow


class StandardsChatCLI:
    """Interactive CLI for standards Q&A."""

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.history = []
        self.question_count = 0

    def print_banner(self):
        """Print welcome banner."""
        print("="*80)
        print("STANDARDS RAG SYSTEM - Interactive Q&A")
        print("="*80)
        print("\nAsk questions about industrial instrumentation standards.")
        print("\nAvailable Standards:")
        print("  - Safety and Protection (SIL ratings, ATEX zones, certifications)")
        print("  - Pressure Measurement")
        print("  - Temperature Measurement")
        print("  - Flow Measurement")
        print("  - Level Measurement")
        print("  - Control Systems")
        print("  - Valves and Actuators")
        print("  - Calibration and Maintenance")
        print("  - Communication and Signals")
        print("  - Condition Monitoring")
        print("  - Analytical Instrumentation")
        print("  - Accessories and Calibration")
        print("\nCommands:")
        print("  - Type your question to get an answer")
        print("  - 'quit' or 'exit' or 'q' - End the session")
        print("  - 'help' - Show this help message")
        print("  - 'history' - Show Q&A history for this session")
        print("  - 'save' - Save session history to file")
        print("\n" + "="*80 + "\n")

    def run(self):
        """Main interaction loop."""
        self.print_banner()

        while True:
            # Get question
            try:
                question = input("\n[Q] Your question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting...")
                self.save_history()
                break

            # Handle commands
            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nExiting...")
                self.save_history()
                break

            if question.lower() == 'help':
                self.print_banner()
                continue

            if question.lower() == 'history':
                self.show_history()
                continue

            if question.lower() == 'save':
                self.save_history(force=True)
                continue

            # Process question
            self.question_count += 1
            print(f"\n[Searching standards documents...]")

            try:
                result = run_standards_rag_workflow(
                    question=question,
                    session_id=self.session_id,
                    top_k=5
                )

                # Display response
                self.display_response(question, result)

                # Save to history
                self.history.append({
                    'question_number': self.question_count,
                    'question': question,
                    'response': result,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                print(f"\n[ERROR] An error occurred: {str(e)}")
                print("Please try rephrasing your question or contact support.")

    def display_response(self, question, result):
        """Display formatted response."""
        if result.get('status') != 'success':
            print(f"\n[ERROR] {result.get('error', 'Unknown error')}")
            return

        data = result['final_response']

        # Answer
        print("\n" + "="*80)
        print("[A] Answer:")
        print("="*80)
        print(data['answer'])

        # Citations
        if data.get('citations'):
            print("\n" + "-"*80)
            print("[Citations]")
            print("-"*80)
            for i, cite in enumerate(data['citations'], 1):
                source = cite.get('source', 'unknown')
                relevance = cite.get('relevance', 0)
                content = cite.get('content', '')

                print(f"\n  [{i}] {source}")
                print(f"      Relevance: {relevance:.2f}")
                if content:
                    # Limit content preview
                    preview = content[:150] + "..." if len(content) > 150 else content
                    print(f"      \"{preview}\"")

        # Metadata
        metadata = data.get('metadata', {})
        print("\n" + "-"*80)
        print("[Metadata]")
        print("-"*80)
        print(f"  Confidence:        {data.get('confidence', 0):.2f}")
        print(f"  Sources used:      {', '.join(data.get('sources_used', []))}")
        print(f"  Processing time:   {metadata.get('processing_time_ms', 0)}ms")
        print(f"  Validation score:  {metadata.get('validation_score', 0):.2f}")
        print(f"  Documents retrieved: {metadata.get('documents_retrieved', 0)}")

        if metadata.get('retry_count', 0) > 0:
            print(f"  Retries:           {metadata.get('retry_count', 0)}")

        if metadata.get('warning'):
            print(f"\n  [WARNING] {metadata.get('warning')}")

        print("="*80)

    def show_history(self):
        """Show Q&A history for this session."""
        if not self.history:
            print("\n[INFO] No questions asked yet in this session.")
            return

        print("\n" + "="*80)
        print("SESSION HISTORY")
        print("="*80)
        print(f"Session ID: {self.session_id}")
        print(f"Total questions: {len(self.history)}")
        print("\n")

        for item in self.history:
            q_num = item['question_number']
            question = item['question']
            response = item['response']
            timestamp = item['timestamp']

            # Parse timestamp
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime('%H:%M:%S')
            except:
                time_str = timestamp

            print(f"[{q_num}] {time_str}")
            print(f"Q: {question}")

            if response.get('status') == 'success':
                data = response['final_response']
                answer_preview = data['answer'][:100] + "..." if len(data['answer']) > 100 else data['answer']
                confidence = data.get('confidence', 0)
                print(f"A: {answer_preview}")
                print(f"   Confidence: {confidence:.2f}")
            else:
                print(f"A: [ERROR] {response.get('error', 'Unknown error')}")

            print()

        print("="*80)

    def save_history(self, force=False):
        """Save Q&A history to file."""
        if not self.history:
            if force:
                print("\n[INFO] No history to save.")
            return

        try:
            filename = f"standards_qa_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # Prepare history for JSON serialization
            history_to_save = {
                'session_id': self.session_id,
                'total_questions': len(self.history),
                'created_at': datetime.now().isoformat(),
                'questions': []
            }

            for item in self.history:
                # Simplify response for storage
                response = item['response']
                if response.get('status') == 'success':
                    data = response['final_response']
                    simplified_response = {
                        'answer': data.get('answer', ''),
                        'confidence': data.get('confidence', 0),
                        'sources_used': data.get('sources_used', []),
                        'metadata': data.get('metadata', {})
                    }
                else:
                    simplified_response = {
                        'error': response.get('error', 'Unknown error')
                    }

                history_to_save['questions'].append({
                    'question_number': item['question_number'],
                    'question': item['question'],
                    'timestamp': item['timestamp'],
                    'response': simplified_response
                })

            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, indent=2, ensure_ascii=False)

            print(f"\n[OK] Session history saved to: {filename}")
            print(f"     Total questions: {len(self.history)}")

        except Exception as e:
            print(f"\n[ERROR] Failed to save history: {str(e)}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        cli = StandardsChatCLI()
        cli.run()
    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
