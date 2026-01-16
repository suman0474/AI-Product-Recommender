"""
PDF Validation Utility
Validates PDF files before processing to ensure quality and prevent errors
"""
import logging
import os
import io
from typing import Dict, Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class PDFValidationError(Exception):
    """Exception raised when PDF validation fails"""
    pass


class PDFValidator:
    """
    Validator for PDF files with comprehensive quality checks

    Validates:
    - PDF signature (%PDF header)
    - File size (1 KB - 50 MB)
    - File extension
    - Content readability
    """

    # Constants
    MIN_FILE_SIZE = 1024  # 1 KB
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    VALID_EXTENSIONS = {'.pdf'}
    PDF_SIGNATURE = b'%PDF-'

    def __init__(self):
        """Initialize PDF validator"""
        self.validation_stats = {
            "total_validated": 0,
            "passed": 0,
            "failed": 0,
            "errors": {}
        }

    def validate_pdf_bytes(
        self,
        pdf_bytes: bytes,
        filename: Optional[str] = None,
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Validate PDF from bytes

        Args:
            pdf_bytes: PDF file content as bytes
            filename: Optional filename for better error messages
            strict: If True, raise exception on validation failure

        Returns:
            Validation result dictionary with status and details

        Raises:
            PDFValidationError: If strict=True and validation fails
        """
        self.validation_stats["total_validated"] += 1
        result = {
            "valid": False,
            "filename": filename or "unknown",
            "size_bytes": len(pdf_bytes),
            "errors": [],
            "warnings": []
        }

        # Check 1: File size
        if len(pdf_bytes) < self.MIN_FILE_SIZE:
            error = f"File too small ({len(pdf_bytes)} bytes, min: {self.MIN_FILE_SIZE})"
            result["errors"].append(error)
            logger.warning(f"[PDF_VALIDATOR] {error} - {filename}")
            self._track_error("file_too_small")

        elif len(pdf_bytes) > self.MAX_FILE_SIZE:
            error = f"File too large ({len(pdf_bytes)} bytes, max: {self.MAX_FILE_SIZE})"
            result["errors"].append(error)
            logger.warning(f"[PDF_VALIDATOR] {error} - {filename}")
            self._track_error("file_too_large")

        # Check 2: PDF signature
        if not pdf_bytes.startswith(self.PDF_SIGNATURE):
            # Check if signature exists anywhere in first 1024 bytes (some PDFs have leading whitespace)
            if self.PDF_SIGNATURE not in pdf_bytes[:1024]:
                error = "Missing PDF signature (%PDF- header)"
                result["errors"].append(error)
                logger.warning(f"[PDF_VALIDATOR] {error} - {filename}")
                self._track_error("missing_signature")
            else:
                result["warnings"].append("PDF signature found after leading bytes (non-standard)")

        # Check 3: File extension (if filename provided)
        if filename:
            ext = os.path.splitext(filename.lower())[1]
            if ext not in self.VALID_EXTENSIONS:
                warning = f"Non-standard file extension: {ext}"
                result["warnings"].append(warning)
                logger.debug(f"[PDF_VALIDATOR] {warning} - {filename}")

        # Check 4: Basic content validation
        try:
            # Try to find common PDF objects
            if b'/Type' not in pdf_bytes and b'/Page' not in pdf_bytes:
                result["warnings"].append("No PDF objects detected (may be empty or corrupted)")
        except Exception as e:
            result["warnings"].append(f"Content validation error: {str(e)}")

        # Determine if valid
        result["valid"] = len(result["errors"]) == 0

        if result["valid"]:
            self.validation_stats["passed"] += 1
            logger.debug(f"[PDF_VALIDATOR] ✓ Valid PDF - {filename} ({len(pdf_bytes)} bytes)")
        else:
            self.validation_stats["failed"] += 1
            logger.warning(f"[PDF_VALIDATOR] ✗ Invalid PDF - {filename}: {', '.join(result['errors'])}")

            if strict:
                raise PDFValidationError(
                    f"PDF validation failed for {filename}: {', '.join(result['errors'])}"
                )

        return result

    def validate_pdf_file(
        self,
        file_path: str,
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Validate PDF from file path

        Args:
            file_path: Path to PDF file
            strict: If True, raise exception on validation failure

        Returns:
            Validation result dictionary

        Raises:
            PDFValidationError: If strict=True and validation fails
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        filename = os.path.basename(file_path)

        try:
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()
            return self.validate_pdf_bytes(pdf_bytes, filename=filename, strict=strict)

        except Exception as e:
            logger.error(f"[PDF_VALIDATOR] Error reading PDF file {file_path}: {e}")
            if strict:
                raise PDFValidationError(f"Failed to read PDF file: {e}")
            return {
                "valid": False,
                "filename": filename,
                "errors": [f"Failed to read file: {str(e)}"],
                "warnings": []
            }

    def validate_pdf_url(self, url: str) -> Dict[str, Any]:
        """
        Validate PDF from URL (basic URL validation only, doesn't download)

        Args:
            url: URL to PDF file

        Returns:
            Validation result dictionary
        """
        result = {
            "valid": True,
            "url": url,
            "errors": [],
            "warnings": []
        }

        try:
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                result["errors"].append(f"Invalid URL scheme: {parsed.scheme}")
                result["valid"] = False

            # Check path extension
            path_lower = parsed.path.lower()
            if not any(path_lower.endswith(ext) for ext in self.VALID_EXTENSIONS):
                result["warnings"].append(f"URL doesn't end with .pdf extension")

            # Check URL length (some URLs are too long and likely broken)
            if len(url) > 2048:
                result["warnings"].append(f"Very long URL ({len(url)} chars)")

        except Exception as e:
            result["errors"].append(f"URL parsing error: {str(e)}")
            result["valid"] = False

        return result

    def batch_validate(
        self,
        pdf_list: list,
        validation_type: str = 'bytes'
    ) -> Dict[str, Any]:
        """
        Validate multiple PDFs and return summary

        Args:
            pdf_list: List of PDFs (bytes, file paths, or URLs depending on validation_type)
            validation_type: Type of validation ('bytes', 'file', or 'url')

        Returns:
            Summary dictionary with results for each PDF
        """
        results = {
            "total": len(pdf_list),
            "valid": 0,
            "invalid": 0,
            "items": []
        }

        for i, pdf_item in enumerate(pdf_list):
            try:
                if validation_type == 'bytes':
                    result = self.validate_pdf_bytes(pdf_item, filename=f"pdf_{i}")
                elif validation_type == 'file':
                    result = self.validate_pdf_file(pdf_item)
                elif validation_type == 'url':
                    result = self.validate_pdf_url(pdf_item)
                else:
                    result = {"valid": False, "errors": [f"Unknown validation type: {validation_type}"]}

                results["items"].append(result)

                if result["valid"]:
                    results["valid"] += 1
                else:
                    results["invalid"] += 1

            except Exception as e:
                logger.error(f"[PDF_VALIDATOR] Error validating PDF {i}: {e}")
                results["items"].append({
                    "valid": False,
                    "errors": [f"Validation exception: {str(e)}"]
                })
                results["invalid"] += 1

        logger.info(
            f"[PDF_VALIDATOR] Batch validation complete: "
            f"{results['valid']}/{results['total']} valid, "
            f"{results['invalid']}/{results['total']} invalid"
        )

        return results

    def _track_error(self, error_type: str):
        """Track error type in statistics"""
        if error_type not in self.validation_stats["errors"]:
            self.validation_stats["errors"][error_type] = 0
        self.validation_stats["errors"][error_type] += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics

        Returns:
            Dictionary with validation stats
        """
        pass_rate = (
            self.validation_stats["passed"] / self.validation_stats["total_validated"] * 100
            if self.validation_stats["total_validated"] > 0
            else 0
        )

        return {
            **self.validation_stats,
            "pass_rate": f"{pass_rate:.1f}%"
        }

    def reset_stats(self):
        """Reset validation statistics"""
        self.validation_stats = {
            "total_validated": 0,
            "passed": 0,
            "failed": 0,
            "errors": {}
        }


# Singleton instance
_validator = None


def get_pdf_validator() -> PDFValidator:
    """
    Get singleton PDF validator instance

    Returns:
        PDFValidator instance
    """
    global _validator
    if _validator is None:
        _validator = PDFValidator()
        logger.info("[PDF_VALIDATOR] Initialized PDF validator")
    return _validator


# Convenience functions
def validate_pdf_bytes(pdf_bytes: bytes, filename: Optional[str] = None, strict: bool = False) -> Dict[str, Any]:
    """Validate PDF from bytes (convenience function)"""
    return get_pdf_validator().validate_pdf_bytes(pdf_bytes, filename=filename, strict=strict)


def validate_pdf_file(file_path: str, strict: bool = False) -> Dict[str, Any]:
    """Validate PDF from file path (convenience function)"""
    return get_pdf_validator().validate_pdf_file(file_path, strict=strict)


def validate_pdf_url(url: str) -> Dict[str, Any]:
    """Validate PDF URL (convenience function)"""
    return get_pdf_validator().validate_pdf_url(url)
