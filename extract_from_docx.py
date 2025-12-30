"""
Script to extract instrumentation standards content directly from DOCX files
and upload to Azure Blob Storage
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from docx import Document
from azure.storage.blob import BlobServiceClient, ContainerClient, ContentSettings
from azure_blob_config import azure_blob_manager


class DocxStandardsExtractor:
    def __init__(self, standards_dir: str):
        self.standards_dir = Path(standards_dir)
        self.extracted_standards = {}

    def extract_text_from_docx(self, docx_file_path: Path) -> dict:
        """Extract meaningful text content from DOCX file"""
        print(f"\nProcessing: {docx_file_path.name}")

        try:
            doc = Document(docx_file_path)

            # Extract all text from paragraphs
            all_text = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text and len(text) > 3:  # Skip very short lines
                    all_text.append(text)

            # Join all paragraphs
            full_text = '\n'.join(all_text)

            # Extract structured information
            standard_info = self.parse_standard_content(full_text, docx_file_path.stem)

            return standard_info

        except Exception as e:
            print(f"  [ERROR] Failed to process {docx_file_path.name}: {e}")
            return None

    def parse_standard_content(self, text: str, filename: str) -> dict:
        """Parse and structure the standards content"""

        # Determine standard type from filename
        standard_type = filename.replace('instrumentation_', '').replace('_standards', '')

        standard_data = {
            'standard_type': standard_type,
            'filename': filename,
            'sections': [],
            'standards_references': [],
            'key_requirements': []
        }

        # Extract standards references (e.g., IEC, ISO, ANSI, ISA patterns)
        standards_patterns = [
            r'IEC\s+\d+[-\d]*',
            r'ISO\s+\d+[-\d]*',
            r'ANSI[/\s]+\w+[-\d.]*',
            r'ISA[/\s-]+\d+[-.\d]*',
            r'ASME\s+\w+[-\d.]*',
            r'API\s+\d+[A-Z]*',
            r'NEMA\s+\w+[-\d.]*',
            r'IEEE\s+\d+[-.\d]*',
            r'EN\s+\d+[-\d]*',
            r'BS\s+\d+[-\d]*',
            r'DIN\s+\d+[-\d]*'
        ]

        found_standards = set()
        for pattern in standards_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                found_standards.add(match.group(0))

        standard_data['standards_references'] = sorted(list(found_standards))

        # Try to identify sections (look for common headers)
        section_patterns = [
            r'(?m)^[\d.]+\s+[A-Z][^\n]{10,100}$',  # Numbered sections
            r'(?m)^[A-Z][A-Z\s]{5,50}:?\s*$',       # ALL CAPS headers
        ]

        sections = []
        for pattern in section_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                sections.append({
                    'title': match.group(0).strip(),
                    'position': match.start()
                })

        # Sort sections by position
        sections.sort(key=lambda x: x['position'])

        # Extract text between sections
        for i, section in enumerate(sections[:20]):  # Limit to first 20 sections
            start_pos = section['position']
            end_pos = sections[i + 1]['position'] if i + 1 < len(sections) else len(text)
            section_text = text[start_pos:end_pos].strip()

            # Limit section text length for storage
            if len(section_text) > 5000:
                section_text = section_text[:5000] + "... [truncated]"

            standard_data['sections'].append({
                'title': section['title'],
                'content': section_text[:1000]  # First 1000 chars as preview
            })

        # Extract key requirements (lines with "shall", "must", "required")
        requirement_patterns = [
            r'[^.]*(?:shall|must|required|mandatory)[^.]*\.',
        ]

        requirements = set()
        for pattern in requirement_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                req = match.group(0).strip()
                if 20 < len(req) < 500:  # Reasonable length
                    requirements.add(req)

        standard_data['key_requirements'] = list(requirements)[:50]  # Limit to 50

        return standard_data

    def process_all_standards(self):
        """Process all DOCX files in the standards directory"""
        docx_files = list(self.standards_dir.glob("instrumentation_*.docx"))

        if not docx_files:
            print(f"\n[WARNING] No instrumentation_*.docx files found in {self.standards_dir}")
            return self.extracted_standards

        print(f"\nFound {len(docx_files)} DOCX files to process")
        print("=" * 80)

        for docx_file in docx_files:
            standard_data = self.extract_text_from_docx(docx_file)

            if standard_data:
                standard_name = standard_data['filename']
                self.extracted_standards[standard_name] = standard_data

                print(f"  [OK] Extracted: {standard_data['standard_type']}")
                print(f"    - Standards referenced: {len(standard_data['standards_references'])}")
                print(f"    - Key requirements: {len(standard_data['key_requirements'])}")
                print(f"    - Sections: {len(standard_data['sections'])}")

        return self.extracted_standards

    def save_to_json(self, output_path: str):
        """Save extracted standards to JSON file"""
        output_file = Path(output_path)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.extracted_standards, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Saved to: {output_file}")
        print(f"  File size: {output_file.stat().st_size:,} bytes")

        return output_file

    def create_summary_report(self):
        """Create a summary report of all standards"""
        summary = {
            'total_standards': len(self.extracted_standards),
            'standards_overview': []
        }

        for name, data in self.extracted_standards.items():
            summary['standards_overview'].append({
                'type': data['standard_type'],
                'filename': data['filename'],
                'standards_referenced': data['standards_references'][:10]  # Top 10
            })

        # Aggregate all unique standards references
        all_refs = set()
        for data in self.extracted_standards.values():
            all_refs.update(data['standards_references'])

        summary['all_unique_standards_referenced'] = sorted(list(all_refs))
        summary['total_unique_standards'] = len(all_refs)

        return summary


class AzureBlobUploader:
    def __init__(self):
        """Initialize Azure Blob Storage client"""
        try:
            if azure_blob_manager.is_available:
                self.blob_service_client = azure_blob_manager.blob_service_client
                print("\n[OK] Connected to Azure Blob Storage")
            else:
                raise Exception("Azure Blob Storage is not available")
        except Exception as e:
            print(f"\n[ERROR] Failed to connect to Azure Blob Storage: {e}")
            self.blob_service_client = None

    def create_container_folder(self, container_name: str, folder_name: str):
        """Create a folder structure in Azure Blob"""
        try:
            container_client = self.blob_service_client.get_container_client(container_name)

            try:
                container_client.get_container_properties()
                print(f"[OK] Using existing container: {container_name}")
            except:
                container_client = self.blob_service_client.create_container(container_name)
                print(f"[OK] Created new container: {container_name}")

            return container_client, folder_name

        except Exception as e:
            print(f"[ERROR] Error creating container/folder: {e}")
            return None, None

    def upload_json_to_blob(self, container_client: ContainerClient,
                           folder_name: str, filename: str, data: dict):
        """Upload JSON data to Azure Blob Storage"""
        try:
            # Create blob name with folder prefix
            blob_name = f"{folder_name}/{filename}"

            # Convert to JSON string
            json_data = json.dumps(data, indent=2, ensure_ascii=False)

            # Upload
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(
                json_data,
                overwrite=True,
                content_settings=ContentSettings(content_type='application/json')
            )

            print(f"[OK] Uploaded: {blob_name} ({len(json_data):,} bytes)")
            return blob_name

        except Exception as e:
            print(f"[ERROR] Error uploading {filename}: {e}")
            return None


def main():
    print("=" * 80)
    print("INSTRUMENTATION STANDARDS EXTRACTION FROM DOCX FILES")
    print("=" * 80)

    # Configuration
    STANDARDS_DIR = r"D:\Standards"
    OUTPUT_DIR = Path(__file__).parent
    CONTAINER_NAME = "instrumentation-standards"
    FOLDER_NAME = f"standards-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Step 1: Extract standards from DOCX
    print("\n[STEP 1] Extracting standards from DOCX files...")
    extractor = DocxStandardsExtractor(STANDARDS_DIR)
    extracted_data = extractor.process_all_standards()

    if not extracted_data:
        print("\n[ERROR] No data extracted. Exiting.")
        return

    # Step 2: Create summary
    print("\n[STEP 2] Creating summary report...")
    summary = extractor.create_summary_report()

    print(f"\n[SUMMARY]")
    print(f"  - Total standards processed: {summary['total_standards']}")
    print(f"  - Unique standards referenced: {summary['total_unique_standards']}")
    standards_list = ', '.join(summary['all_unique_standards_referenced'][:20])
    print(f"  - Standards (sample): {standards_list}")

    # Step 3: Save locally
    print("\n[STEP 3] Saving to local JSON files...")

    full_data_file = OUTPUT_DIR / "instrumentation_standards_full.json"
    extractor.save_to_json(str(full_data_file))

    summary_file = OUTPUT_DIR / "instrumentation_standards_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved summary to: {summary_file}")

    # Step 4: Upload to Azure Blob Storage
    print("\n[STEP 4] Uploading to Azure Blob Storage...")
    uploader = AzureBlobUploader()

    if uploader.blob_service_client:
        container_client, folder = uploader.create_container_folder(
            CONTAINER_NAME, FOLDER_NAME
        )

        if container_client:
            print(f"\n[OK] Created folder structure: {CONTAINER_NAME}/{folder}")

            # Upload full data
            uploader.upload_json_to_blob(
                container_client, folder,
                "instrumentation_standards_full.json",
                extracted_data
            )

            # Upload summary
            uploader.upload_json_to_blob(
                container_client, folder,
                "instrumentation_standards_summary.json",
                summary
            )

            # Upload individual standard files
            print("\n[OK] Uploading individual standard files...")
            for name, data in extracted_data.items():
                uploader.upload_json_to_blob(
                    container_client, folder,
                    f"{data['standard_type']}.json",
                    data
                )

            print(f"\n" + "=" * 80)
            print(f"[SUCCESS] COMPLETED SUCCESSFULLY")
            print(f"=" * 80)
            print(f"\nAzure Blob Location:")
            print(f"  Container: {CONTAINER_NAME}")
            print(f"  Folder: {folder}")
            print(f"\nLocal files saved to: {OUTPUT_DIR}")
    else:
        print("\n[WARNING] Skipping Azure upload - no connection available")
        print("  Local files have been saved successfully")


if __name__ == "__main__":
    # Install required packages if needed
    try:
        from docx import Document
    except ImportError:
        print("Installing required package: python-docx")
        os.system("pip install python-docx")
        from docx import Document

    main()
