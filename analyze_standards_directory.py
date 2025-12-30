"""
Script to analyze all files in D:\Standards directory
Walks through subdirectories, reads files, and provides analysis
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any

class DirectoryAnalyzer:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.analysis_report = {
            'total_files': 0,
            'total_directories': 0,
            'files_by_extension': {},
            'file_details': []
        }

    def is_text_file(self, file_path: Path) -> bool:
        """Check if file is likely a text file"""
        text_extensions = {
            '.txt', '.md', '.py', '.json', '.xml', '.html', '.css', '.js',
            '.csv', '.yaml', '.yml', '.ini', '.cfg', '.conf', '.log',
            '.rst', '.tex', '.sql', '.sh', '.bat', '.ps1', '.c', '.cpp',
            '.h', '.java', '.cs', '.php', '.rb', '.go', '.rs', '.ts',
            '.jsx', '.tsx', '.vue', '.r', '.m', '.swift', '.kt'
        }
        return file_path.suffix.lower() in text_extensions

    def analyze_file_content(self, file_path: Path) -> Dict[str, Any]:
        """Analyze individual file content"""
        file_info = {
            'path': str(file_path),
            'name': file_path.name,
            'extension': file_path.suffix,
            'size_bytes': 0,
            'line_count': 0,
            'content_analysis': '',
            'error': None
        }

        try:
            file_info['size_bytes'] = file_path.stat().st_size

            if not self.is_text_file(file_path):
                file_info['content_analysis'] = f"Binary or non-text file ({file_path.suffix})"
                return file_info

            # Read and analyze text file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    file_info['line_count'] = len(lines)

                    # Analyze content
                    analysis = self.analyze_text_content(lines, file_path)
                    file_info['content_analysis'] = analysis

            except UnicodeDecodeError:
                # Try with different encoding
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        lines = f.readlines()
                        file_info['line_count'] = len(lines)
                        analysis = self.analyze_text_content(lines, file_path)
                        file_info['content_analysis'] = analysis
                except Exception as e:
                    file_info['content_analysis'] = f"Could not decode file: {str(e)}"

        except Exception as e:
            file_info['error'] = str(e)
            file_info['content_analysis'] = f"Error reading file: {str(e)}"

        return file_info

    def analyze_text_content(self, lines: List[str], file_path: Path) -> str:
        """Provide detailed analysis of text content"""
        analysis_parts = []

        # File type specific analysis
        if file_path.suffix == '.py':
            analysis_parts.append("Python source code file")
            self.analyze_python_content(lines, analysis_parts)
        elif file_path.suffix == '.json':
            analysis_parts.append("JSON data file")
            self.analyze_json_content(file_path, analysis_parts)
        elif file_path.suffix in ['.md', '.txt']:
            analysis_parts.append("Text/Documentation file")
            self.analyze_text_document(lines, analysis_parts)
        elif file_path.suffix in ['.xml', '.html']:
            analysis_parts.append("Markup language file")
            self.analyze_markup_content(lines, analysis_parts)
        elif file_path.suffix in ['.csv']:
            analysis_parts.append("CSV data file")
            self.analyze_csv_content(lines, analysis_parts)
        elif file_path.suffix in ['.yaml', '.yml']:
            analysis_parts.append("YAML configuration file")
            self.analyze_yaml_content(lines, analysis_parts)
        else:
            analysis_parts.append(f"Text file with extension {file_path.suffix}")
            self.analyze_generic_text(lines, analysis_parts)

        # General statistics
        non_empty_lines = [l for l in lines if l.strip()]
        analysis_parts.append(f"\n  Total lines: {len(lines)}")
        analysis_parts.append(f"  Non-empty lines: {len(non_empty_lines)}")
        analysis_parts.append(f"  Empty lines: {len(lines) - len(non_empty_lines)}")

        # Show first few lines for context
        if lines:
            analysis_parts.append("\n  First few lines:")
            for i, line in enumerate(lines[:5], 1):
                preview = line.rstrip()[:100]
                if len(line.rstrip()) > 100:
                    preview += "..."
                analysis_parts.append(f"    Line {i}: {preview}")

        return '\n'.join(analysis_parts)

    def analyze_python_content(self, lines: List[str], analysis_parts: List[str]):
        """Analyze Python code"""
        imports = [l for l in lines if l.strip().startswith(('import ', 'from '))]
        classes = [l for l in lines if l.strip().startswith('class ')]
        functions = [l for l in lines if l.strip().startswith('def ')]
        comments = [l for l in lines if l.strip().startswith('#')]

        analysis_parts.append(f"  - Import statements: {len(imports)}")
        analysis_parts.append(f"  - Class definitions: {len(classes)}")
        analysis_parts.append(f"  - Function definitions: {len(functions)}")
        analysis_parts.append(f"  - Comment lines: {len(comments)}")

    def analyze_json_content(self, file_path: Path, analysis_parts: List[str]):
        """Analyze JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    analysis_parts.append(f"  - JSON object with {len(data)} top-level keys")
                    analysis_parts.append(f"  - Keys: {', '.join(list(data.keys())[:10])}")
                elif isinstance(data, list):
                    analysis_parts.append(f"  - JSON array with {len(data)} elements")
        except Exception as e:
            analysis_parts.append(f"  - JSON parsing error: {str(e)}")

    def analyze_text_document(self, lines: List[str], analysis_parts: List[str]):
        """Analyze text/markdown document"""
        headers = [l for l in lines if l.strip().startswith('#')]
        bullet_points = [l for l in lines if l.strip().startswith(('-', '*', '+'))]

        if headers:
            analysis_parts.append(f"  - Markdown headers: {len(headers)}")
        if bullet_points:
            analysis_parts.append(f"  - Bullet points: {len(bullet_points)}")

    def analyze_markup_content(self, lines: List[str], analysis_parts: List[str]):
        """Analyze HTML/XML content"""
        tags = [l for l in lines if '<' in l and '>' in l]
        analysis_parts.append(f"  - Lines with tags: {len(tags)}")

    def analyze_csv_content(self, lines: List[str], analysis_parts: List[str]):
        """Analyze CSV content"""
        if lines:
            header = lines[0].strip()
            columns = header.split(',')
            analysis_parts.append(f"  - Columns: {len(columns)}")
            analysis_parts.append(f"  - Data rows: {len(lines) - 1}")

    def analyze_yaml_content(self, lines: List[str], analysis_parts: List[str]):
        """Analyze YAML content"""
        keys = [l for l in lines if ':' in l and not l.strip().startswith('#')]
        analysis_parts.append(f"  - YAML key-value pairs: {len(keys)}")

    def analyze_generic_text(self, lines: List[str], analysis_parts: List[str]):
        """Generic text analysis"""
        total_chars = sum(len(l) for l in lines)
        words = sum(len(l.split()) for l in lines)
        analysis_parts.append(f"  - Total characters: {total_chars}")
        analysis_parts.append(f"  - Total words: {words}")

    def walk_directory(self):
        """Walk through directory and analyze all files"""
        print(f"\n{'='*80}")
        print(f"ANALYZING DIRECTORY: {self.root_path}")
        print(f"{'='*80}\n")

        if not self.root_path.exists():
            print(f"ERROR: Path does not exist: {self.root_path}")
            return

        # Walk through directory
        for root, dirs, files in os.walk(self.root_path):
            self.analysis_report['total_directories'] += len(dirs)

            current_path = Path(root)
            relative_path = current_path.relative_to(self.root_path)

            if files:
                print(f"\n{'-'*80}")
                print(f"DIRECTORY: {relative_path if str(relative_path) != '.' else 'Root'}")
                print(f"{'-'*80}")

            for file in files:
                file_path = current_path / file
                self.analysis_report['total_files'] += 1

                # Track extensions
                ext = file_path.suffix.lower() or 'no_extension'
                self.analysis_report['files_by_extension'][ext] = \
                    self.analysis_report['files_by_extension'].get(ext, 0) + 1

                # Analyze file
                print(f"\nFILE: {file}")
                print(f"   Path: {file_path}")

                file_info = self.analyze_file_content(file_path)
                self.analysis_report['file_details'].append(file_info)

                print(f"   Size: {file_info['size_bytes']:,} bytes")
                print(f"\n   ANALYSIS:")
                for line in file_info['content_analysis'].split('\n'):
                    print(f"   {line}")

                if file_info['error']:
                    print(f"   WARNING ERROR: {file_info['error']}")

    def print_summary(self):
        """Print summary statistics"""
        print(f"\n\n{'='*80}")
        print("SUMMARY REPORT")
        print(f"{'='*80}")
        print(f"\nTotal Directories: {self.analysis_report['total_directories']}")
        print(f"Total Files: {self.analysis_report['total_files']}")
        print(f"\nFiles by Extension:")
        for ext, count in sorted(self.analysis_report['files_by_extension'].items(),
                                 key=lambda x: x[1], reverse=True):
            print(f"  {ext}: {count}")

    def save_report(self, output_file: str):
        """Save detailed report to JSON file"""
        report_path = Path(output_file)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_report, f, indent=2)
        print(f"\n\nDetailed report saved to: {report_path}")


def main():
    # Target directory
    standards_path = r"D:\Standards"

    # Create analyzer
    analyzer = DirectoryAnalyzer(standards_path)

    # Analyze directory
    analyzer.walk_directory()

    # Print summary
    analyzer.print_summary()

    # Save detailed report
    output_path = Path(__file__).parent / "standards_analysis_report.json"
    analyzer.save_report(str(output_path))

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
