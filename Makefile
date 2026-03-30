# NeuroScan — Makefile
# Autor: Diego L. Silva (github.com/diegslva)
# Uso: make <target>   |   make help

.DEFAULT_GOAL := help

TARGET_LINUX_X64   := x86_64-unknown-linux-gnu
TARGET_LINUX_ARM64 := aarch64-unknown-linux-gnu

.PHONY: help check lint lint-fix format format-check test test-v build build-release run \
        linux-x64 linux-arm64 linux-all all clippy fmt fmt-check clean

help: ## Lista todos os targets disponiveis
	@powershell -NoProfile -Command "Get-Content Makefile | Select-String '^\w.*:.*##' | ForEach-Object { $$line = $$_.Line; $$parts = $$line -split '##'; $$target = ($$parts[0] -split ':')[0].Trim(); $$desc = $$parts[1].Trim(); Write-Host ('  {0,-18} {1}' -f $$target, $$desc) }"

# ── Qualidade ─────────────────────────────────────────────────────────────────

check: format-check lint test ## Suite completa de qualidade (fmt + clippy + tests)

lint: ## cargo clippy — warnings como erros
	cargo clippy --all-targets --all-features -- -D warnings

lint-fix: ## cargo clippy com auto-fix
	cargo clippy --all-targets --all-features --fix --allow-dirty

format: ## cargo fmt — formata todo o codigo
	cargo fmt --all

format-check: ## cargo fmt --check — verifica formatacao sem alterar
	cargo fmt --all -- --check

# Aliases usados em dev cotidiano
fmt: format ## Alias: cargo fmt
fmt-check: format-check ## Alias: cargo fmt --check
clippy: lint ## Alias: cargo clippy

# ── Testes ────────────────────────────────────────────────────────────────────

test: ## cargo test (lib + integration tests, sem GPU)
	cargo test --lib --tests

test-v: ## cargo test com output detalhado
	cargo test --lib --tests -- --nocapture

# ── Build ─────────────────────────────────────────────────────────────────────

build: ## cargo build (debug, Windows nativo)
	cargo build

build-release: ## cargo build --release com LTO
	cargo build --release

# Alias para compatibilidade com Makefile anterior
release: build-release ## Alias: cargo build --release

run: ## Executa o viewer em modo debug
	powershell -NoProfile -ExecutionPolicy Bypass -Command "$$env:RUST_LOG='debug'; cargo run"

# ── Cross-compile ─────────────────────────────────────────────────────────────

linux-x64: ## Cross-compile release Linux x86_64
	cross build --release --target $(TARGET_LINUX_X64)

linux-arm64: ## Cross-compile release Linux ARM64
	cross build --release --target $(TARGET_LINUX_ARM64)

linux-all: linux-x64 linux-arm64 ## Cross-compile todos os targets Linux

all: build-release linux-all ## Build release para todos os targets (Windows + Linux)

# ── Limpeza ───────────────────────────────────────────────────────────────────

clean: ## Remove artefatos de build
	cargo clean
