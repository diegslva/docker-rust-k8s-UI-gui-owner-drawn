param(
    [Parameter(Position = 0)]
    [ValidateSet(
        "build", "release", "run",
        "linux-x64", "linux-arm64", "linux-all", "all",
        "check", "clippy", "fmt", "fmt-check",
        "clean", "help"
    )]
    [string]$Action = "help"
)

$ErrorActionPreference = "Stop"

$TARGET_LINUX_X64 = "x86_64-unknown-linux-gnu"
$TARGET_LINUX_ARM64 = "aarch64-unknown-linux-gnu"

function Invoke-Build {
    Write-Host "[build] Compilando debug (Windows nativo)..." -ForegroundColor Cyan
    cargo build
}

function Invoke-Release {
    Write-Host "[release] Compilando release (Windows nativo)..." -ForegroundColor Cyan
    cargo build --release
}

function Invoke-Run {
    Write-Host "[run] Executando com RUST_LOG=debug..." -ForegroundColor Cyan
    $env:RUST_LOG = "debug"
    cargo run
}

function Invoke-LinuxX64 {
    Write-Host "[linux-x64] Cross-compile release Linux x86_64..." -ForegroundColor Cyan
    cross build --release --target $TARGET_LINUX_X64
}

function Invoke-LinuxArm64 {
    Write-Host "[linux-arm64] Cross-compile release Linux ARM64..." -ForegroundColor Cyan
    cross build --release --target $TARGET_LINUX_ARM64
}

function Invoke-LinuxAll {
    Invoke-LinuxX64
    Invoke-LinuxArm64
}

function Invoke-All {
    Invoke-Release
    Invoke-LinuxAll
}

function Invoke-Check {
    Write-Host "[check] Verificando compilacao..." -ForegroundColor Cyan
    cargo check
}

function Invoke-Clippy {
    Write-Host "[clippy] Lint com clippy..." -ForegroundColor Cyan
    cargo clippy -- -D warnings
}

function Invoke-Fmt {
    Write-Host "[fmt] Formatando codigo..." -ForegroundColor Cyan
    cargo fmt
}

function Invoke-FmtCheck {
    Write-Host "[fmt-check] Verificando formatacao..." -ForegroundColor Cyan
    cargo fmt --check
}

function Invoke-Clean {
    Write-Host "[clean] Limpando artefatos de build..." -ForegroundColor Cyan
    cargo clean
}

function Show-Help {
    Write-Host ""
    Write-Host "  docker-rust-k8s-ui-gui - dev.ps1" -ForegroundColor Yellow
    Write-Host "  =================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Build:" -ForegroundColor Green
    Write-Host "    build          Build debug (Windows nativo)"
    Write-Host "    release        Build release (Windows nativo)"
    Write-Host "    run            Executar em modo debug com RUST_LOG=debug"
    Write-Host ""
    Write-Host "  Cross-compile:" -ForegroundColor Green
    Write-Host "    linux-x64      Cross-compile release Linux x86_64"
    Write-Host "    linux-arm64    Cross-compile release Linux ARM64"
    Write-Host "    linux-all      Cross-compile todos os targets Linux"
    Write-Host "    all            Build release para todos os targets"
    Write-Host ""
    Write-Host "  Qualidade:" -ForegroundColor Green
    Write-Host "    check          Verificar compilacao sem gerar binario"
    Write-Host "    clippy         Lint com clippy (warnings = erro)"
    Write-Host "    fmt            Formatar codigo com rustfmt"
    Write-Host "    fmt-check      Verificar formatacao sem alterar"
    Write-Host ""
    Write-Host "  Limpeza:" -ForegroundColor Green
    Write-Host "    clean          Limpar artefatos de build"
    Write-Host ""
}

switch ($Action) {
    "build"      { Invoke-Build }
    "release"    { Invoke-Release }
    "run"        { Invoke-Run }
    "linux-x64"  { Invoke-LinuxX64 }
    "linux-arm64" { Invoke-LinuxArm64 }
    "linux-all"  { Invoke-LinuxAll }
    "all"        { Invoke-All }
    "check"      { Invoke-Check }
    "clippy"     { Invoke-Clippy }
    "fmt"        { Invoke-Fmt }
    "fmt-check"  { Invoke-FmtCheck }
    "clean"      { Invoke-Clean }
    "help"       { Show-Help }
}
