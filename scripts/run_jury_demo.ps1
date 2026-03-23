param(
    [ValidateSet("compare", "peak", "gui")]
    [string]$Mode = "compare"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = ".deps;."

switch ($Mode) {
    "compare" {
        python main.py compare
    }
    "peak" {
        python scripts/run_peak_demo.py
    }
    "gui" {
        python gui_app.py
    }
}

