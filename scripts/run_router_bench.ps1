# Cold-KV router benchmark: recreate sims (+ ray-serve), then GuideLLM.
#
# Usage (repo root, PowerShell):
#   .\scripts\run_router_bench.ps1 -Router prefix
#   .\scripts\run_router_bench.ps1 -Router p2c
#   .\scripts\run_router_bench.ps1 -Router round_robin -Rate 3
#   .\scripts\run_router_bench.ps1 -Router consistent_hash -Dataset prefix_shared
#
# Always force-recreates sim-0..7 so each run starts with an empty KV cache.
# Sims use per-token TTFT from docker-compose.yml.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("prefix", "p2c", "round_robin", "consistent_hash")]
    [string]$Router,

    [string]$Dataset = "prefix_shared",

    [double]$Rate = 3,

    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $RunId) {
    $RunId = $Router
}

switch ($Router) {
    "prefix" {
        $env:SERVE_SCRIPT = "src/prefix_router.py"
        $env:ROUTER = "p2c"  # ignored by prefix_router.py
    }
    default {
        $env:SERVE_SCRIPT = "src/router.py"
        $env:ROUTER = $Router
    }
}

$services = @(
    "sim-0", "sim-1", "sim-2", "sim-3", "sim-4", "sim-5", "sim-6", "sim-7", "ray-serve"
)

Write-Host "SERVE_SCRIPT=$($env:SERVE_SCRIPT) ROUTER=$($env:ROUTER)"
Write-Host "Cold recreate: $($services -join ', ')"
& docker compose up -d --force-recreate @services
if ($LASTEXITCODE -ne 0) {
    throw "docker compose up failed (exit $LASTEXITCODE)"
}

Write-Host "Waiting for ray-serve healthy..."
$deadline = (Get-Date).AddMinutes(5)
do {
    $health = docker inspect msc_onlab1_ray-ray-serve-1 --format "{{.State.Health.Status}}" 2>$null
    if ($health -eq "healthy") {
        break
    }
    if ((Get-Date) -gt $deadline) {
        throw "ray-serve not healthy within timeout (status=$health)"
    }
    Start-Sleep -Seconds 2
} while ($true)

Write-Host "GuideLLM: dataset=$Dataset run-id=$RunId rate=$Rate"
& docker compose run --rm guidellm --dataset $Dataset --run-id $RunId --rate $Rate
if ($LASTEXITCODE -ne 0) {
    throw "guidellm failed (exit $LASTEXITCODE)"
}

Write-Host "Done -> results/$RunId/guidellm.json (+ assembled_chat_completions.json if capture ok)"
