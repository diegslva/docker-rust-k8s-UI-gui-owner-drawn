//! Auto-setup do ambiente Python para inferencia ONNX.
//!
//! Estrategia de deteccao (prioridade):
//! 1. Venv local ja configurado (`neuroscan_data/venv/`) — usa direto
//! 2. Python do sistema com deps instaladas — usa direto
//! 3. Python do sistema sem deps — cria venv e instala
//! 4. Sem Python — baixa python-build-standalone, cria venv, instala deps
//!
//! Cross-platform: Windows x64, Linux x86_64, Linux aarch64 (Jetson).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use tracing::{info, warn};

/// Versao minima do Python suportada.
const MIN_PYTHON_VERSION: (u32, u32) = (3, 10);

/// Release do python-build-standalone para download.
const PBS_RELEASE: &str = "20250317";
const PBS_PYTHON_VERSION: &str = "3.10.16";

/// Dependencias Python necessarias para a inferencia.
const REQUIRED_PACKAGES: &[&str] = &["nibabel", "onnxruntime", "scipy", "skimage"];

/// Pacotes pip a instalar (nomes de pacote pip, nao de import).
const PIP_INSTALL_PACKAGES: &[&str] = &["nibabel", "onnxruntime", "scipy", "scikit-image", "numpy"];

/// Ambiente Python pronto para uso.
#[derive(Debug, Clone)]
pub(crate) struct PythonEnv {
    /// Path do executavel Python (sistema ou venv).
    pub python_bin: PathBuf,
    /// Fonte do Python (para logging).
    pub source: PythonSource,
    /// Versao do Python detectado (major, minor).
    pub version: (u32, u32),
}

impl PythonEnv {
    /// Log de resumo visivel — chamado apos setup bem-sucedido.
    ///
    /// Mostra: versao Python, fonte (sistema/venv/standalone), path, e providers ONNX.
    pub fn log_summary(&self) {
        let providers = self.detect_ort_providers();
        let msg = format!(
            "[NeuroScan] Python {}.{} ({}) -- {}",
            self.version.0,
            self.version.1,
            self.source,
            self.python_bin.display(),
        );
        eprintln!("{msg}");
        info!("{msg}");

        let prov_msg = format!("[NeuroScan] ONNX Runtime providers: {providers}");
        eprintln!("{prov_msg}");
        info!("{prov_msg}");
    }

    /// Detecta quais execution providers o ONNX Runtime suporta neste Python.
    fn detect_ort_providers(&self) -> String {
        let output = Command::new(self.python_bin.as_os_str())
            .args([
                "-c",
                "import onnxruntime as ort; print(','.join(ort.get_available_providers()))",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output();

        match output {
            Ok(out) if out.status.success() => {
                String::from_utf8_lossy(&out.stdout).trim().to_string()
            }
            _ => "nao detectado".to_string(),
        }
    }
}

/// Origem do Python detectado.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PythonSource {
    /// Python do sistema com todas as deps.
    System,
    /// Venv local (neuroscan_data/venv/).
    LocalVenv,
    /// Python standalone baixado + venv.
    Standalone,
}

impl std::fmt::Display for PythonSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PythonSource::System => write!(f, "sistema"),
            PythonSource::LocalVenv => write!(f, "venv local"),
            PythonSource::Standalone => write!(f, "standalone"),
        }
    }
}

/// Callback para reportar progresso do setup.
pub(crate) type SetupProgressFn = Box<dyn Fn(&str) + Send>;

/// Garante que um ambiente Python funcional esta disponivel.
///
/// Tenta na ordem: venv local > Python sistema > download standalone.
/// Retorna o path do executavel Python pronto para uso.
pub(crate) fn ensure_python_env(progress: Option<SetupProgressFn>) -> Result<PythonEnv, String> {
    let report = |msg: &str| {
        if let Some(ref f) = progress {
            f(msg);
        }
    };

    let base_dir = neuroscan_data_dir();

    // 1. Venv local ja configurado?
    let venv_python = venv_python_path(&base_dir);
    if venv_python.exists() && check_deps(&venv_python) {
        let version = get_python_version(&venv_python.to_string_lossy()).unwrap_or((3, 10));
        info!(python = %venv_python.display(), "venv local com deps encontrado");
        let env = PythonEnv {
            python_bin: venv_python,
            source: PythonSource::LocalVenv,
            version,
        };
        env.log_summary();
        return Ok(env);
    }

    // 2. Python do sistema com deps?
    if let Some(system_python) = find_system_python() {
        let version = get_python_version(&system_python).unwrap_or((3, 10));

        if check_deps_str(&system_python) {
            info!(python = %system_python, "Python do sistema com deps encontrado");
            let env = PythonEnv {
                python_bin: PathBuf::from(&system_python),
                source: PythonSource::System,
                version,
            };
            env.log_summary();
            return Ok(env);
        }

        // 3. Python do sistema sem deps — criar venv e instalar
        info!("Python do sistema encontrado mas deps faltando — criando venv");
        report("Criando ambiente Python...");
        create_venv(&system_python, &base_dir)?;
        report("Instalando dependencias (nibabel, onnxruntime, scipy, scikit-image)...");
        install_deps(&venv_python)?;

        if check_deps(&venv_python) {
            info!(python = %venv_python.display(), "venv local criado com sucesso");
            let env = PythonEnv {
                python_bin: venv_python,
                source: PythonSource::LocalVenv,
                version,
            };
            env.log_summary();
            return Ok(env);
        }
        warn!("venv criado mas deps nao verificadas — tentando standalone");
    }

    // 4. Sem Python adequado — baixar standalone
    eprintln!("[NeuroScan] Python nao encontrado no sistema — baixando standalone...");
    report("Baixando Python portavel (~40MB)...");
    let standalone_python = download_standalone_python(&base_dir)?;

    report("Criando ambiente Python...");
    let standalone_str = standalone_python.to_string_lossy().to_string();
    create_venv(&standalone_str, &base_dir)?;

    report("Instalando dependencias (nibabel, onnxruntime, scipy, scikit-image)...");
    install_deps(&venv_python)?;

    if check_deps(&venv_python) {
        let version = get_python_version(&venv_python.to_string_lossy()).unwrap_or((
            PBS_PYTHON_VERSION
                .split('.')
                .next()
                .unwrap_or("3")
                .parse()
                .unwrap_or(3),
            PBS_PYTHON_VERSION
                .split('.')
                .nth(1)
                .unwrap_or("10")
                .parse()
                .unwrap_or(10),
        ));
        info!(python = %venv_python.display(), "Python standalone + venv pronto");
        let env = PythonEnv {
            python_bin: venv_python,
            source: PythonSource::Standalone,
            version,
        };
        env.log_summary();
        return Ok(env);
    }

    Err("Falha ao configurar ambiente Python apos download standalone".to_string())
}

/// Diretorio base para dados do NeuroScan (ao lado do executavel).
fn neuroscan_data_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("neuroscan_data")))
        .unwrap_or_else(|| PathBuf::from("neuroscan_data"))
}

/// Path do Python dentro do venv local.
fn venv_python_path(base_dir: &Path) -> PathBuf {
    let venv_dir = base_dir.join("venv");
    if cfg!(target_os = "windows") {
        venv_dir.join("Scripts").join("python.exe")
    } else {
        venv_dir.join("bin").join("python3")
    }
}

/// Detecta Python no sistema com versao >= MIN_PYTHON_VERSION.
fn find_system_python() -> Option<String> {
    let candidates: &[&str] = if cfg!(target_os = "windows") {
        &["python", "python3"]
    } else {
        &["python3", "python"]
    };

    for cmd in candidates {
        if let Some(version) = get_python_version(cmd) {
            if version >= MIN_PYTHON_VERSION {
                info!(python = %cmd, version = ?version, "Python do sistema adequado");
                return Some(cmd.to_string());
            }
            warn!(python = %cmd, version = ?version, min = ?MIN_PYTHON_VERSION, "Python do sistema com versao incompativel");
        }
    }
    None
}

/// Retorna a versao (major, minor) de um executavel Python.
fn get_python_version(cmd: &str) -> Option<(u32, u32)> {
    let output = Command::new(cmd)
        .args([
            "-c",
            "import sys; print(sys.version_info.major, sys.version_info.minor)",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() == 2 {
        let major = parts[0].parse::<u32>().ok()?;
        let minor = parts[1].parse::<u32>().ok()?;
        Some((major, minor))
    } else {
        None
    }
}

/// Verifica se todas as deps necessarias estao instaladas (por Path).
fn check_deps(python_bin: &Path) -> bool {
    check_deps_str(&python_bin.to_string_lossy())
}

/// Verifica se todas as deps necessarias estao instaladas (por nome de comando).
fn check_deps_str(python: &str) -> bool {
    let check_script = REQUIRED_PACKAGES
        .iter()
        .map(|p| format!("import {p}"))
        .collect::<Vec<_>>()
        .join("; ");

    let output = Command::new(python)
        .args(["-c", &check_script])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output();

    matches!(output, Ok(out) if out.status.success())
}

/// Cria um venv no neuroscan_data/venv/.
///
/// No Linux aarch64 (Jetson), usa `--system-site-packages` para herdar
/// o onnxruntime-gpu da NVIDIA instalado globalmente.
fn create_venv(python: &str, base_dir: &Path) -> Result<(), String> {
    let venv_dir = base_dir.join("venv");
    std::fs::create_dir_all(base_dir).map_err(|e| format!("falha ao criar diretorio: {e}"))?;

    let mut args = vec!["-m", "venv"];

    // Jetson (aarch64 Linux): herdar pacotes do sistema (onnxruntime-gpu NVIDIA)
    let use_system_packages = cfg!(target_os = "linux") && cfg!(target_arch = "aarch64");
    if use_system_packages {
        args.push("--system-site-packages");
        info!("Jetson detectado (aarch64) — venv com --system-site-packages");
    }

    let venv_str = venv_dir.to_string_lossy().to_string();
    args.push(&venv_str);

    let output = Command::new(python)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("falha ao executar python -m venv: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("falha ao criar venv: {stderr}"));
    }

    info!(venv = %venv_dir.display(), "venv criado");
    Ok(())
}

/// Instala dependencias via pip no venv.
fn install_deps(venv_python: &Path) -> Result<(), String> {
    let mut args = vec![
        "-m".to_string(),
        "pip".to_string(),
        "install".to_string(),
        "--quiet".to_string(),
        "--upgrade".to_string(),
    ];
    for pkg in PIP_INSTALL_PACKAGES {
        args.push(pkg.to_string());
    }

    info!(packages = ?PIP_INSTALL_PACKAGES, "instalando dependencias pip");

    let output = Command::new(venv_python.as_os_str())
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("falha ao executar pip install: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("pip install falhou: {stderr}"));
    }

    info!("dependencias pip instaladas com sucesso");
    Ok(())
}

/// Baixa e extrai python-build-standalone para neuroscan_data/python/.
fn download_standalone_python(base_dir: &Path) -> Result<PathBuf, String> {
    let python_dir = base_dir.join("python");

    // Se ja existe, retorna o path do executavel
    let python_bin = standalone_python_bin(&python_dir);
    if python_bin.exists() {
        info!(python = %python_bin.display(), "Python standalone ja presente");
        return Ok(python_bin);
    }

    std::fs::create_dir_all(base_dir).map_err(|e| format!("falha ao criar diretorio: {e}"))?;

    let url = standalone_download_url();
    let archive_path = base_dir.join("python-standalone.tar.gz");

    info!(url = %url, "baixando Python standalone");

    // Download via curl (built-in no Windows 10+ e Linux)
    let curl_cmd = if cfg!(target_os = "windows") {
        "curl.exe"
    } else {
        "curl"
    };

    let output = Command::new(curl_cmd)
        .args(["-fSL", "-o"])
        .arg(archive_path.as_os_str())
        .arg(&url)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("falha ao executar curl: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("download falhou: {stderr}"));
    }

    info!(
        size = archive_path.metadata().map(|m| m.len()).unwrap_or(0),
        "download concluido"
    );

    // Extrair via tar (built-in no Windows 10+ e Linux)
    std::fs::create_dir_all(&python_dir)
        .map_err(|e| format!("falha ao criar diretorio python: {e}"))?;

    let output = Command::new("tar")
        .args(["xzf"])
        .arg(archive_path.as_os_str())
        .arg("-C")
        .arg(python_dir.as_os_str())
        .arg("--strip-components=1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("falha ao extrair tar: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("extracao falhou: {stderr}"));
    }

    // Limpar archive
    let _ = std::fs::remove_file(&archive_path);

    let python_bin = standalone_python_bin(&python_dir);
    if python_bin.exists() {
        info!(python = %python_bin.display(), "Python standalone extraido");
        Ok(python_bin)
    } else {
        Err(format!(
            "Python standalone nao encontrado em {}",
            python_bin.display()
        ))
    }
}

/// Path do executavel Python dentro do diretorio standalone.
fn standalone_python_bin(python_dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        python_dir.join("python.exe")
    } else {
        python_dir.join("bin").join("python3")
    }
}

/// URL de download do python-build-standalone para a plataforma atual.
fn standalone_download_url() -> String {
    let arch = if cfg!(target_os = "windows") && cfg!(target_arch = "x86_64") {
        "x86_64-pc-windows-msvc"
    } else if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
        "x86_64-unknown-linux-gnu"
    } else if cfg!(target_os = "linux") && cfg!(target_arch = "aarch64") {
        "aarch64-unknown-linux-gnu"
    } else {
        // Fallback para Linux x86_64
        "x86_64-unknown-linux-gnu"
    };

    format!(
        "https://github.com/indygreg/python-build-standalone/releases/download/{tag}/cpython-{ver}%2B{tag}-{arch}-install_only.tar.gz",
        tag = PBS_RELEASE,
        ver = PBS_PYTHON_VERSION,
        arch = arch,
    )
}
