.DEFAULT_GOAL := help

TARGET_LINUX_X64  := x86_64-unknown-linux-gnu
TARGET_LINUX_ARM64 := aarch64-unknown-linux-gnu

## Build

.PHONY: build
build: ## Build debug (Windows nativo)
	cargo build

.PHONY: release
release: ## Build release (Windows nativo)
	cargo build --release

.PHONY: run
run: ## Executar em modo debug com RUST_LOG=debug
	powershell -NoProfile -ExecutionPolicy Bypass -Command "$$env:RUST_LOG='debug'; cargo run"

## Cross-compile

.PHONY: linux-x64
linux-x64: ## Cross-compile release Linux x86_64
	cross build --release --target $(TARGET_LINUX_X64)

.PHONY: linux-arm64
linux-arm64: ## Cross-compile release Linux ARM64
	cross build --release --target $(TARGET_LINUX_ARM64)

.PHONY: linux-all
linux-all: linux-x64 linux-arm64 ## Cross-compile todos os targets Linux

.PHONY: all
all: release linux-all ## Build release para todos os targets (Windows + Linux)

## Qualidade

.PHONY: check
check: ## Verificar compilacao sem gerar binario
	cargo check

.PHONY: clippy
clippy: ## Lint com clippy (warnings = erro)
	cargo clippy -- -D warnings

.PHONY: fmt
fmt: ## Formatar codigo com rustfmt
	cargo fmt

.PHONY: fmt-check
fmt-check: ## Verificar formatacao sem alterar
	cargo fmt --check

## Limpeza

.PHONY: clean
clean: ## Limpar artefatos de build
	cargo clean

## Ajuda

.PHONY: help
help: ## Listar todos os targets disponiveis
	@echo.
	@echo   Targets disponiveis:
	@echo   ====================
	@powershell -NoProfile -Command "Get-Content Makefile | Select-String '^\w.*:.*##' | ForEach-Object { $$line = $$_.Line; $$parts = $$line -split '##'; $$target = ($$parts[0] -split ':')[0].Trim(); $$desc = $$parts[1].Trim(); Write-Host ('  {0,-16} {1}' -f $$target, $$desc) }"
	@echo.
