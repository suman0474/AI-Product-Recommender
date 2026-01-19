#!/bin/bash

# FIX #17 Master Test Runner
# Executes all tests for 60+ specification implementation

echo "======================================================================="
echo "FIX #17 MASTER TEST SUITE"
echo "60+ Specification Enrichment - Complete Testing"
echo "======================================================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_file="$2"

    echo ""
    echo -e "${BLUE}Running: ${test_name}${NC}"
    echo "File: ${test_file}"
    echo "---"

    if [ -f "${test_file}" ]; then
        python "${test_file}"
        local exit_code=$?

        TOTAL_TESTS=$((TOTAL_TESTS + 1))

        if [ $exit_code -eq 0 ]; then
            echo -e "${GREEN}✅ ${test_name} PASSED${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo -e "${RED}❌ ${test_name} FAILED${NC}"
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    else
        echo -e "${RED}❌ Test file not found: ${test_file}${NC}"
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Run all tests
echo -e "${BLUE}Starting test execution...${NC}\n"

run_test "FIX #17 Core Implementation Tests" "test_fix17_implementation.py"
run_test "FIX #17 PPI Workflow Integration Tests" "test_ppi_fix17_integration.py"

# Print final summary
echo ""
echo "======================================================================="
echo "FINAL TEST SUMMARY"
echo "======================================================================="
echo ""
echo "Total Test Suites Run: ${TOTAL_TESTS}"
echo -e "${GREEN}Passed: ${PASSED_TESTS}${NC}"
echo -e "${RED}Failed: ${FAILED_TESTS}${NC}"

if [ ${FAILED_TESTS} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ALL TESTS PASSED ✅                                       ║${NC}"
    echo -e "${GREEN}║  FIX #17 is ready for deployment!                          ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  SOME TESTS FAILED ❌                                      ║${NC}"
    echo -e "${RED}║  Please fix issues before deployment                       ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    exit 1
fi
