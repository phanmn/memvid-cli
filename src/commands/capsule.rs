use std::ffi::OsStr;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{ArgAction, Parser};

#[cfg(unix)]
use libc::{isatty, tcgetattr, tcsetattr, termios, ECHO, TCSANOW};

#[derive(Parser, Debug)]
#[command(about = "Encrypt a memory file, creating an encrypted capsule (.mv2e)")]
pub struct LockArgs {
    /// Path to the .mv2 file to encrypt
    #[arg(value_name = "FILE")]
    pub file: PathBuf,

    /// Interactive password prompt (default, safest)
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "password_stdin")]
    pub password: bool,

    /// Read password from stdin (for CI/scripts)
    #[arg(long = "password-stdin", action = ArgAction::SetTrue, conflicts_with = "password")]
    pub password_stdin: bool,

    /// Output file path (default: <FILE>.mv2e)
    #[arg(short, long, value_name = "PATH")]
    pub out: Option<PathBuf>,

    /// Overwrite output file if it exists
    #[arg(long, action = ArgAction::SetTrue)]
    pub force: bool,

    /// Keep the original .mv2 file after encryption (default: delete for security)
    #[arg(long = "keep-original", action = ArgAction::SetTrue)]
    pub keep_original: bool,

    /// Output result as JSON
    #[arg(long, action = ArgAction::SetTrue)]
    pub json: bool,
}

#[derive(Parser, Debug)]
#[command(about = "Decrypt an encrypted capsule, recreating the original .mv2 file")]
pub struct UnlockArgs {
    /// Path to the .mv2e file to decrypt
    #[arg(value_name = "FILE")]
    pub file: PathBuf,

    /// Interactive password prompt (default, safest)
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "password_stdin")]
    pub password: bool,

    /// Read password from stdin (for CI/scripts)
    #[arg(long = "password-stdin", action = ArgAction::SetTrue, conflicts_with = "password")]
    pub password_stdin: bool,

    /// Output file path (default: <FILE> without .mv2e)
    #[arg(short, long, value_name = "PATH")]
    pub out: Option<PathBuf>,

    /// Overwrite output file if it exists
    #[arg(long, action = ArgAction::SetTrue)]
    pub force: bool,

    /// Output result as JSON
    #[arg(long, action = ArgAction::SetTrue)]
    pub json: bool,
}

pub fn handle_lock(args: LockArgs) -> Result<()> {
    ensure_extension(&args.file, "mv2")?;

    let output_path = args.out.unwrap_or_else(|| args.file.with_extension("mv2e"));
    ensure_distinct_paths(&args.file, &output_path)?;
    ensure_output_path(&output_path, args.force)?;

    let mut password_bytes = read_password_bytes(
        PasswordMode::from_args(args.password, args.password_stdin),
        true,
    )?;

    let input_len = fs::metadata(&args.file)
        .with_context(|| format!("failed to stat {}", args.file.display()))?
        .len();

    let result_path = memvid_core::encryption::lock_file(
        &args.file,
        Some(output_path.as_path()),
        &password_bytes,
    )
    .map_err(anyhow::Error::from)?;
    password_bytes.fill(0);

    // Delete the original .mv2 file for security (unless --keep-original)
    if !args.keep_original {
        fs::remove_file(&args.file)
            .with_context(|| format!("failed to remove original file {}", args.file.display()))?;

        if !args.json {
            println!(
                "Deleted: {} (use --keep-original to preserve)",
                args.file.display()
            );
        }
    }

    print_capsule_result(
        &args.file,
        &result_path,
        input_len,
        args.json,
        !args.keep_original,
    )?;
    Ok(())
}

pub fn handle_unlock(args: UnlockArgs) -> Result<()> {
    ensure_extension(&args.file, "mv2e")?;

    let output_path = args.out.unwrap_or_else(|| args.file.with_extension("mv2"));
    ensure_distinct_paths(&args.file, &output_path)?;
    ensure_output_path(&output_path, args.force)?;

    let mut password_bytes = read_password_bytes(
        PasswordMode::from_args(args.password, args.password_stdin),
        false,
    )?;

    let input_len = fs::metadata(&args.file)
        .with_context(|| format!("failed to stat {}", args.file.display()))?
        .len();

    let result_path = memvid_core::encryption::unlock_file(
        &args.file,
        Some(output_path.as_path()),
        &password_bytes,
    )
    .map_err(anyhow::Error::from)?;
    password_bytes.fill(0);

    print_capsule_result(&args.file, &result_path, input_len, args.json, false)?;
    Ok(())
}

fn print_capsule_result(
    input: &Path,
    output: &Path,
    size: u64,
    json: bool,
    deleted_original: bool,
) -> Result<()> {
    if json {
        let payload = serde_json::json!({
            "input": input.display().to_string(),
            "output": output.display().to_string(),
            "size": size,
            "original_deleted": deleted_original,
        });
        println!("{}", payload);
        return Ok(());
    }

    println!("Wrote: {}", output.display());
    Ok(())
}

fn ensure_extension(path: &Path, expected: &str) -> Result<()> {
    let ext = path.extension().and_then(OsStr::to_str);
    if ext != Some(expected) {
        bail!(
            "Expected .{} file, got: {}",
            expected,
            ext.unwrap_or("<none>")
        );
    }
    Ok(())
}

fn ensure_distinct_paths(input: &Path, output: &Path) -> Result<()> {
    if input == output {
        bail!("Refusing to overwrite input file: {}", input.display());
    }
    Ok(())
}

fn ensure_output_path(output: &Path, force: bool) -> Result<()> {
    if !output.exists() {
        return Ok(());
    }
    if !force {
        bail!(
            "Output file already exists: {}\nUse --force to overwrite",
            output.display()
        );
    }
    fs::remove_file(output).with_context(|| format!("failed to remove {}", output.display()))?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum PasswordMode {
    Prompt,
    Stdin,
}

impl PasswordMode {
    fn from_args(_password: bool, password_stdin: bool) -> Self {
        if password_stdin {
            return PasswordMode::Stdin;
        }
        PasswordMode::Prompt
    }
}

fn read_password_bytes(mode: PasswordMode, confirm: bool) -> Result<Vec<u8>> {
    let password = match mode {
        PasswordMode::Stdin => read_password_from_stdin()?,
        PasswordMode::Prompt => read_password_from_prompt(confirm)?,
    };

    if password.trim().is_empty() {
        bail!("Password cannot be empty");
    }

    Ok(password.into_bytes())
}

fn read_password_from_stdin() -> Result<String> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut line = String::new();
    let bytes = reader
        .read_line(&mut line)
        .context("failed to read password from stdin")?;
    if bytes == 0 {
        bail!("Password cannot be empty");
    }
    Ok(line.trim_end_matches(&['\n', '\r'][..]).to_string())
}

fn read_password_from_prompt(confirm: bool) -> Result<String> {
    let password = read_password_hidden("Password: ")?;
    if !confirm {
        return Ok(password);
    }
    let confirm_pw = read_password_hidden("Confirm:  ")?;
    if password != confirm_pw {
        bail!("Passwords do not match");
    }
    Ok(password)
}

fn read_password_hidden(prompt: &str) -> Result<String> {
    let is_tty = stdin_is_tty();
    let mut stderr = io::stderr();
    stderr.write_all(prompt.as_bytes())?;
    stderr.flush()?;

    let guard = if is_tty { disable_stdin_echo()? } else { None };

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut line = String::new();
    reader.read_line(&mut line)?;

    if guard.is_some() {
        let _ = stderr.write_all(b"\n");
        let _ = stderr.flush();
    }

    Ok(line.trim_end_matches(&['\n', '\r'][..]).to_string())
}

fn stdin_is_tty() -> bool {
    #[cfg(unix)]
    unsafe {
        isatty(libc::STDIN_FILENO) == 1
    }

    #[cfg(not(unix))]
    {
        false
    }
}

#[cfg(unix)]
fn disable_stdin_echo() -> Result<Option<EchoGuard>> {
    let fd = libc::STDIN_FILENO;
    unsafe {
        let mut current: termios = std::mem::zeroed();
        if tcgetattr(fd, &mut current) != 0 {
            return Ok(None);
        }
        let mut updated = current;
        updated.c_lflag &= !ECHO;
        if tcsetattr(fd, TCSANOW, &updated) != 0 {
            return Ok(None);
        }
        Ok(Some(EchoGuard {
            fd,
            previous: current,
        }))
    }
}

#[cfg(not(unix))]
fn disable_stdin_echo() -> Result<Option<EchoGuard>> {
    Ok(None)
}

#[cfg(unix)]
struct EchoGuard {
    fd: i32,
    previous: termios,
}

#[cfg(unix)]
impl Drop for EchoGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = tcsetattr(self.fd, TCSANOW, &self.previous);
        }
    }
}

#[cfg(not(unix))]
struct EchoGuard;

#[cfg(not(unix))]
impl Drop for EchoGuard {
    fn drop(&mut self) {}
}
