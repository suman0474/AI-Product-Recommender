# Bulk update script to replace Gemini model versions
# This script replaces:
# - gemini-2.0-flash-exp -> gemini-2.5-flash
# - gemini-2.0-flash -> gemini-2.5-flash
# - gemini-1.5-flash -> gemini-2.5-flash

# Set working directory to script location
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Array of files to update (excluding venv)
$filesToUpdate = @(
    "advanced_parameters.py",
    "agentic/api.py",
    "agentic/deep_agent/llm_specs_generator.py",
    "agentic/deep_agent/schema_populator.py",
    "agentic/deep_agent/standards_deep_agent.py",
    "agentic/deep_agent/user_specs_extractor.py",
    "agentic/deep_agent/workflow.py",
    "agentic/grounded_chat_workflow.py",
    "agentic/index_rag/index_rag_agent.py",
    "agentic/index_rag/index_rag_memory.py",
    "agentic/instrument_identifier_workflow.py",
    "agentic/potential_product_index.py",
    "agentic/product_info_orchestrator.py",
    "agentic/rag_components.py",
    "agentic/shared_agents.py",
    "agentic/solution_workflow.py",
    "agentic/standards_rag/standards_chat_agent.py",
    "agentic/standards_rag/standards_rag_workflow.py",
    "agentic/strategy_rag/strategy_chat_agent.py",
    "agentic/workflow_orchestrator.py",
    "diagnose_llm.py",
    "llm_fallback.py",
    "llm_standardization.py",
    "loading.py",
    "main.py",
    "pdf_processor.py",
    "pdf_utils.py",
    "product_search_workflow/ppi_tools.py",
    "test.py",
    "tools/analysis_tools.py",
    "tools/instrument_tools.py",
    "tools/intent_tools.py",
    "tools/metadata_filter.py",
    "tools/parallel_indexer.py",
    "tools/ranking_tools.py",
    "tools/sales_agent_tools.py",
    "tools/sales_workflow_tools.py",
    "tools/schema_tools.py"
)

$totalReplacements = 0
$updatedFiles = @()

Write-Host "Starting Gemini model version updates..." -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

foreach ($file in $filesToUpdate) {
    $filePath = Join-Path $scriptDir $file
    
    if (-Not (Test-Path $filePath)) {
        Write-Host "SKIPPED: $file (file not found)" -ForegroundColor Yellow
        continue
    }
    
    # Read file content
    $content = Get-Content -Path $filePath -Raw
    $originalContent = $content
    
    # Count replacements before making changes
    $countExp = ($content | Select-String -Pattern "gemini-2\.0-flash-exp" -AllMatches).Matches.Count
    $count20 = ($content | Select-String -Pattern "gemini-2\.0-flash(?!-)" -AllMatches).Matches.Count
    $count15 = ($content | Select-String -Pattern "gemini-1\.5-flash" -AllMatches).Matches.Count
    
    $replacementsInFile = 0
    
    # Replace gemini-2.0-flash-exp with gemini-2.5-flash
    if ($countExp -gt 0) {
        $content = $content -replace "gemini-2\.0-flash-exp", "gemini-2.5-flash"
        $replacementsInFile += $countExp
    }
    
    # Replace gemini-2.0-flash with gemini-2.5-flash (not gemini-2.0-flash-exp)
    if ($count20 -gt 0) {
        $content = $content -replace "gemini-2\.0-flash(?!-)", "gemini-2.5-flash"
        $replacementsInFile += $count20
    }
    
    # Replace gemini-1.5-flash with gemini-2.5-flash
    if ($count15 -gt 0) {
        $content = $content -replace "gemini-1\.5-flash", "gemini-2.5-flash"
        $replacementsInFile += $count15
    }
    
    # Only update file if changes were made
    if ($content -ne $originalContent) {
        # Create backup
        $backupPath = $filePath + ".bak"
        Copy-Item -Path $filePath -Destination $backupPath -Force
        
        # Write updated content
        Set-Content -Path $filePath -Value $content -NoNewline
        
        $updatedFiles += $file
        $totalReplacements += $replacementsInFile
        
        Write-Host "UPDATED: $file" -ForegroundColor Green
        Write-Host "  - Replacements made: $replacementsInFile" -ForegroundColor Green
        Write-Host "  - Backup created: ${file}.bak" -ForegroundColor Gray
    }
    else {
        Write-Host "SKIPPED: $file (no matching patterns found)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "Update Summary:" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "Total files updated: $($updatedFiles.Count)" -ForegroundColor Green
Write-Host "Total replacements made: $totalReplacements" -ForegroundColor Green
Write-Host ""

if ($updatedFiles.Count -gt 0) {
    Write-Host "Files updated:" -ForegroundColor Green
    foreach ($file in $updatedFiles) {
        Write-Host "  - $file" -ForegroundColor Green
    }
    Write-Host ""
    Write-Host "Backup files created with .bak extension for each updated file." -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Replacements made:" -ForegroundColor Cyan
Write-Host "  - gemini-2.0-flash-exp -> gemini-2.5-flash" -ForegroundColor Cyan
Write-Host "  - gemini-2.0-flash -> gemini-2.5-flash" -ForegroundColor Cyan
Write-Host "  - gemini-1.5-flash -> gemini-2.5-flash" -ForegroundColor Cyan
Write-Host ""

if ($updatedFiles.Count -eq 0) {
    Write-Host "No files needed updating." -ForegroundColor Yellow
}
else {
    Write-Host "Update completed successfully!" -ForegroundColor Green
}
