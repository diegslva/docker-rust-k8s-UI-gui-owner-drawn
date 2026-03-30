# NeuroScan -- scripts/dev.ps1
# Interface PowerShell para comandos de desenvolvimento no Windows.
# Autor: Diego L. Silva (github.com/diegslva)
#
# Uso: .\scripts\dev.ps1 <comando>

$ErrorActionPreference = "SilentlyContinue"

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$TARGET_LINUX_X64   = "x86_64-unknown-linux-gnu"
$TARGET_LINUX_ARM64 = "aarch64-unknown-linux-gnu"

function Show-Help {
    Write-Host ""
    Write-Host "  NeuroScan dev.ps1 -- comandos disponiveis:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Build:" -ForegroundColor Yellow
    Write-Host "    run           cargo run (viewer em debug)" -ForegroundColor Green
    Write-Host "    build         cargo build (debug)" -ForegroundColor Green
    Write-Host "    release       cargo build --release" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Cross-compile:" -ForegroundColor Yellow
    Write-Host "    linux-x64     Cross-compile release Linux x86_64" -ForegroundColor Green
    Write-Host "    linux-arm64   Cross-compile release Linux ARM64" -ForegroundColor Green
    Write-Host "    linux-all     Cross-compile todos os targets Linux" -ForegroundColor Green
    Write-Host "    all           Build release para todos os targets" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Qualidade:" -ForegroundColor Yellow
    Write-Host "    check         fmt-check + clippy + tests (suite completa)" -ForegroundColor Green
    Write-Host "    test          cargo test --lib --tests" -ForegroundColor Green
    Write-Host "    test-v        cargo test com output detalhado" -ForegroundColor Green
    Write-Host "    fmt           cargo fmt --all" -ForegroundColor Green
    Write-Host "    fmt-check     cargo fmt --check" -ForegroundColor Green
    Write-Host "    lint          cargo clippy -D warnings" -ForegroundColor Green
    Write-Host "    lint-fix      cargo clippy --fix" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Limpeza:" -ForegroundColor Yellow
    Write-Host "    clean         cargo clean" -ForegroundColor Green
    Write-Host ""
}

switch ($Command) {
    "run" {
        Write-Host "[run] Executando com RUST_LOG=debug..." -ForegroundColor Cyan
        $env:RUST_LOG = "debug"
        cargo run
    }
    "build" {
        Write-Host "[build] Compilando debug..." -ForegroundColor Cyan
        cargo build
    }
    "release" {
        Write-Host "[release] Compilando release..." -ForegroundColor Cyan
        cargo build --release
    }
    "linux-x64" {
        Write-Host "[linux-x64] Cross-compile Linux x86_64..." -ForegroundColor Cyan
        cross build --release --target $TARGET_LINUX_X64
    }
    "linux-arm64" {
        Write-Host "[linux-arm64] Cross-compile Linux ARM64..." -ForegroundColor Cyan
        cross build --release --target $TARGET_LINUX_ARM64
    }
    "linux-all" {
        & $PSCommandPath linux-x64
        & $PSCommandPath linux-arm64
    }
    "all" {
        & $PSCommandPath release
        & $PSCommandPath linux-all
    }
    "check" {
        Write-Host "[check] fmt-check..." -ForegroundColor Cyan
        cargo fmt --all -- --check
        if ($LASTEXITCODE -ne 0) { Write-Host "fmt check falhou" -ForegroundColor Red; exit 1 }
        Write-Host "[check] clippy..." -ForegroundColor Cyan
        cargo clippy --all-targets --all-features -- -D warnings
        if ($LASTEXITCODE -ne 0) { Write-Host "clippy falhou" -ForegroundColor Red; exit 1 }
        Write-Host "[check] tests..." -ForegroundColor Cyan
        cargo test --lib --tests
        if ($LASTEXITCODE -ne 0) { Write-Host "testes falharam" -ForegroundColor Red; exit 1 }
        Write-Host "check OK" -ForegroundColor Green
    }
    "test" {
        Write-Host "[test] cargo test --lib --tests..." -ForegroundColor Cyan
        cargo test --lib --tests
    }
    "test-v" {
        Write-Host "[test-v] cargo test com output detalhado..." -ForegroundColor Cyan
        cargo test --lib --tests -- --nocapture
    }
    "fmt" {
        Write-Host "[fmt] Formatando codigo..." -ForegroundColor Cyan
        cargo fmt --all
    }
    "fmt-check" {
        Write-Host "[fmt-check] Verificando formatacao..." -ForegroundColor Cyan
        cargo fmt --all -- --check
    }
    "lint" {
        Write-Host "[lint] cargo clippy -D warnings..." -ForegroundColor Cyan
        cargo clippy --all-targets --all-features -- -D warnings
    }
    "lint-fix" {
        Write-Host "[lint-fix] cargo clippy --fix..." -ForegroundColor Cyan
        cargo clippy --all-targets --all-features --fix --allow-dirty
    }
    "clean" {
        Write-Host "[clean] cargo clean..." -ForegroundColor Cyan
        cargo clean
    }
    default { Show-Help }
}
