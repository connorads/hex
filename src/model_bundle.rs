//! Pinned multi-file model bundles: staged download, whole-bundle
//! verification, and atomic publication beside the single-file GGUF models.
//!
//! A bundle is published only after every pinned file has its exact size and
//! SHA-256. Partial downloads stay in a private staging directory so an
//! interrupted install resumes instead of appearing installed.

use std::fs::{self, File};
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

use color_eyre::eyre::{Result, WrapErr, bail, eyre};
use serde::{Deserialize, Serialize};

use crate::transcription_models::{ModelPreparationStage, modified_ns, sha256_hex};

pub struct BundleFile {
    /// Relative path inside the bundle, using `/` separators.
    pub path: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

pub struct ModelBundle {
    /// Directory name below the models directory. Includes the revision so a
    /// new pin never reuses an older bundle or its compiled caches.
    pub directory: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
    pub files: &'static [BundleFile],
}

impl ModelBundle {
    pub const fn bytes(&self) -> u64 {
        let mut total = 0;
        let mut index = 0;
        while index < self.files.len() {
            total += self.files[index].bytes;
            index += 1;
        }
        total
    }

    pub fn url(&self, file: &BundleFile) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repository, self.revision, file.path
        )
    }

    pub fn path(&self, models: &Path) -> PathBuf {
        models.join(self.directory)
    }

    fn staging_path(&self, models: &Path) -> PathBuf {
        models.join(format!("{}.partial", self.directory))
    }

    fn replaced_path(&self, models: &Path) -> PathBuf {
        models.join(format!("{}.replaced", self.directory))
    }

    fn receipt_path(&self, models: &Path) -> PathBuf {
        models.join(format!("{}.verified.json", self.directory))
    }
}

#[derive(Deserialize, Serialize)]
struct BundleReceipt {
    revision: String,
    files: Vec<VerifiedFile>,
}

#[derive(Deserialize, Serialize)]
struct VerifiedFile {
    path: String,
    bytes: u64,
    sha256: String,
    modified_ns: u64,
}

/// Downloads one bundle file to `output`, resuming any bytes already there,
/// and reports the current size of `output` through `progress`.
pub type Fetch<'a> = &'a dyn Fn(&str, &Path, &AtomicBool, &dyn Fn(u64)) -> Result<()>;

/// Every pinned file is present as a regular file of the pinned size. This is
/// the cheap installed check; [`is_verified`] also requires a matching receipt.
pub fn is_complete(models: &Path, bundle: &ModelBundle) -> bool {
    let root = bundle.path(models);
    bundle
        .files
        .iter()
        .all(|file| regular_file_len(&root, file.path).ok() == Some(file.bytes))
}

pub fn is_verified(models: &Path, bundle: &ModelBundle) -> bool {
    let root = bundle.path(models);
    let Ok(receipt) = fs::read(bundle.receipt_path(models))
        .map_err(color_eyre::Report::from)
        .and_then(|data| serde_json::from_slice::<BundleReceipt>(&data).map_err(Into::into))
    else {
        return false;
    };
    receipt.revision == bundle.revision
        && receipt.files.len() == bundle.files.len()
        && bundle.files.iter().all(|file| {
            let Some(verified) = receipt.files.iter().find(|entry| entry.path == file.path) else {
                return false;
            };
            let Ok(metadata) = regular_file_metadata(&root, file.path) else {
                return false;
            };
            verified.bytes == file.bytes
                && verified.sha256 == file.sha256
                && metadata.len() == file.bytes
                && modified_ns(&metadata) == Some(verified.modified_ns)
        })
}

/// Verifies the published bundle without downloading or replacing anything.
/// The caller holds the models directory download lock.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub fn verify_installed(models: &Path, bundle: &ModelBundle, canceled: &AtomicBool) -> Result<()> {
    if !is_verified(models, bundle) {
        verify_tree(&bundle.path(models), bundle, canceled)?;
        write_receipt(models, bundle)?;
    }
    Ok(())
}

/// Installs `bundle` below `models`. The caller holds the models directory
/// download lock. An existing invalid bundle is replaced only after its
/// replacement has been completely verified.
pub fn install(
    models: &Path,
    bundle: &ModelBundle,
    canceled: &AtomicBool,
    downloaded: &AtomicU64,
    stage: &AtomicU8,
    fetch: Fetch<'_>,
) -> Result<PathBuf> {
    validate_spec(bundle)?;
    let destination = bundle.path(models);
    let total = bundle.bytes();
    check_canceled(canceled)?;
    if destination.exists() {
        if is_verified(models, bundle) {
            downloaded.store(total, Ordering::Relaxed);
            return Ok(destination);
        }
        ModelPreparationStage::Verifying.store(stage);
        match verify_tree(&destination, bundle, canceled) {
            Ok(()) => {
                write_receipt(models, bundle)?;
                downloaded.store(total, Ordering::Relaxed);
                return Ok(destination);
            }
            Err(error) if canceled.load(Ordering::Relaxed) => return Err(error),
            Err(error) => {
                tracing::warn!(%error, path = %destination.display(), "will replace invalid model bundle after verifying its replacement");
            }
        }
    }

    let staging = bundle.staging_path(models);
    create_private_dir(&staging)?;
    let mut completed = 0;
    for file in bundle.files {
        check_canceled(canceled)?;
        let target = safe_join(&staging, file.path)?;
        if let Some(parent) = target.parent() {
            create_private_dir(parent)?;
        }
        reject_symlinks(&staging, file.path)?;
        // A complete-size file from an interrupted run is checked once and kept.
        if regular_file_len(&staging, file.path).ok() == Some(file.bytes) {
            ModelPreparationStage::Verifying.store(stage);
            if verify_file(&target, file, canceled).is_ok() {
                completed += file.bytes;
                downloaded.store(completed, Ordering::Relaxed);
                continue;
            }
            check_canceled(canceled)?;
            tracing::warn!(path = file.path, "replacing invalid staged bundle file");
            fs::remove_file(&target)?;
        } else if fs::symlink_metadata(&target).is_ok_and(|metadata| metadata.len() > file.bytes) {
            fs::remove_file(&target)?;
        }
        ModelPreparationStage::Downloading.store(stage);
        let base = completed;
        fetch(&bundle.url(file), &target, canceled, &|bytes| {
            downloaded.store(base + bytes.min(file.bytes), Ordering::Relaxed);
        })?;
        check_canceled(canceled)?;
        reject_symlinks(&staging, file.path)?;
        ModelPreparationStage::Verifying.store(stage);
        if let Err(error) = verify_file(&target, file, canceled) {
            if !canceled.load(Ordering::Relaxed) {
                let _ = fs::remove_file(&target);
            }
            return Err(error);
        }
        File::open(&target)?.sync_all()?;
        completed += file.bytes;
        downloaded.store(completed, Ordering::Relaxed);
    }
    if !is_complete_tree(&staging, bundle) {
        bail!("model bundle staging is incomplete");
    }
    sync_directories(&staging, bundle)?;

    let replaced = bundle.replaced_path(models);
    remove_path(&replaced)?;
    remove_path(&bundle.receipt_path(models))?;
    if fs::symlink_metadata(&destination).is_ok() {
        fs::rename(&destination, &replaced)?;
    }
    fs::rename(&staging, &destination)?;
    File::open(models)?.sync_all()?;
    write_receipt(models, bundle)?;
    remove_path(&replaced)?;
    downloaded.store(total, Ordering::Relaxed);
    Ok(destination)
}

/// Deletes the published bundle, any compiled caches inside it, its receipt,
/// and any staged or replaced copy. The caller holds the download lock.
pub fn remove(models: &Path, bundle: &ModelBundle) -> Result<()> {
    for path in [
        bundle.receipt_path(models),
        bundle.path(models),
        bundle.staging_path(models),
        bundle.replaced_path(models),
    ] {
        remove_path(&path).wrap_err_with(|| format!("could not delete {}", path.display()))?;
    }
    if models.is_dir() {
        File::open(models)?.sync_all()?;
    }
    Ok(())
}

fn validate_spec(bundle: &ModelBundle) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    if bundle.directory.is_empty()
        || bundle.directory.contains(['/', '\\'])
        || bundle.directory.starts_with('.')
    {
        bail!("unsafe model bundle directory: {}", bundle.directory);
    }
    for file in bundle.files {
        safe_relative(file.path)?;
        if !seen.insert(file.path)
            || file.bytes == 0
            || file.sha256.len() != 64
            || !file
                .sha256
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        {
            bail!("invalid model bundle file: {}", file.path);
        }
    }
    Ok(())
}

fn safe_relative(path: &str) -> Result<&Path> {
    let relative = Path::new(path);
    if path.is_empty()
        || path.contains('\\')
        || !relative
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        bail!("unsafe model bundle path: {path}");
    }
    Ok(relative)
}

fn safe_join(root: &Path, path: &str) -> Result<PathBuf> {
    Ok(root.join(safe_relative(path)?))
}

/// Refuses a symlink at the root or at any component of `path`.
fn reject_symlinks(root: &Path, path: &str) -> Result<()> {
    let mut current = root.to_path_buf();
    if fs::symlink_metadata(&current).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
        bail!("model bundle directory is a symbolic link");
    }
    for component in safe_relative(path)?.components() {
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                bail!("model bundle contains a symbolic link: {path}")
            }
            Ok(_) => {}
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn regular_file_metadata(root: &Path, path: &str) -> Result<fs::Metadata> {
    reject_symlinks(root, path)?;
    let metadata = fs::symlink_metadata(safe_join(root, path)?)?;
    if !metadata.is_file() {
        bail!("model bundle entry is not a regular file: {path}");
    }
    Ok(metadata)
}

fn regular_file_len(root: &Path, path: &str) -> Result<u64> {
    regular_file_metadata(root, path).map(|metadata| metadata.len())
}

fn is_complete_tree(root: &Path, bundle: &ModelBundle) -> bool {
    bundle
        .files
        .iter()
        .all(|file| regular_file_len(root, file.path).ok() == Some(file.bytes))
}

fn verify_file(path: &Path, file: &BundleFile, canceled: &AtomicBool) -> Result<()> {
    let length = fs::symlink_metadata(path)?.len();
    if length != file.bytes {
        bail!(
            "downloaded {length} bytes for {}, expected {}",
            file.path,
            file.bytes
        );
    }
    if sha256_hex(path, Some(canceled))? != file.sha256 {
        bail!("checksum mismatch for {}", file.path);
    }
    Ok(())
}

fn verify_tree(root: &Path, bundle: &ModelBundle, canceled: &AtomicBool) -> Result<()> {
    validate_spec(bundle)?;
    for file in bundle.files {
        check_canceled(canceled)?;
        regular_file_metadata(root, file.path)
            .wrap_err_with(|| format!("model bundle is missing {}", file.path))?;
        verify_file(&safe_join(root, file.path)?, file, canceled)?;
    }
    Ok(())
}

fn write_receipt(models: &Path, bundle: &ModelBundle) -> Result<()> {
    let root = bundle.path(models);
    let files = bundle
        .files
        .iter()
        .map(|file| {
            let metadata = regular_file_metadata(&root, file.path)?;
            Ok(VerifiedFile {
                path: file.path.into(),
                bytes: file.bytes,
                sha256: file.sha256.into(),
                modified_ns: modified_ns(&metadata)
                    .ok_or_else(|| eyre!("model modification time is unavailable"))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    crate::transcription_models::write_json_atomically(
        &bundle.receipt_path(models),
        &BundleReceipt {
            revision: bundle.revision.into(),
            files,
        },
    )
}

fn sync_directories(root: &Path, bundle: &ModelBundle) -> Result<()> {
    let mut directories = bundle
        .files
        .iter()
        .filter_map(|file| Path::new(file.path).parent())
        .flat_map(Path::ancestors)
        .map(|directory| root.join(directory))
        .collect::<Vec<_>>();
    directories.sort();
    directories.dedup();
    // Children before parents so each rename target is durable first.
    for directory in directories.iter().rev() {
        File::open(directory)?.sync_all()?;
    }
    Ok(())
}

fn create_private_dir(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(path)?;
    }
    #[cfg(not(unix))]
    fs::create_dir_all(path)?;
    if fs::symlink_metadata(path)?.file_type().is_symlink() {
        bail!("model bundle directory is a symbolic link");
    }
    Ok(())
}

fn remove_path(path: &Path) -> Result<()> {
    let result = match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        Err(error) => Err(error),
    };
    match result {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn check_canceled(canceled: &AtomicBool) -> Result<()> {
    if canceled.load(Ordering::Relaxed) {
        bail!("model download canceled");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::io::Write;

    use super::*;

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    // sha256("hello"), sha256("world!") and sha256("{}").
    const FILES: &[BundleFile] = &[
        BundleFile {
            path: "model.mlpackage/Data/weights.bin",
            bytes: 5,
            sha256: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
        },
        BundleFile {
            path: "decoder.bin",
            bytes: 6,
            sha256: "711e9609339e92b03ddc0a211827dba421f38f9ed8b9d806e1ffdd8c15ffa03d",
        },
        BundleFile {
            path: "manifest.json",
            bytes: 2,
            sha256: "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        },
    ];
    const BUNDLE: ModelBundle = ModelBundle {
        directory: "fixture-bundle-rev",
        repository: "fixture/repository",
        revision: "rev",
        files: FILES,
    };

    fn contents() -> HashMap<&'static str, &'static [u8]> {
        HashMap::from([
            (
                "https://huggingface.co/fixture/repository/resolve/rev/model.mlpackage/Data/weights.bin",
                b"hello".as_slice(),
            ),
            (
                "https://huggingface.co/fixture/repository/resolve/rev/decoder.bin",
                b"world!".as_slice(),
            ),
            (
                "https://huggingface.co/fixture/repository/resolve/rev/manifest.json",
                b"{}".as_slice(),
            ),
        ])
    }

    fn directory() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "hex-model-bundle-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    /// Appends the remainder of the fixture, like `curl --continue-at -`.
    fn resume_fetch(
        url: &str,
        output: &Path,
        _: &AtomicBool,
        progress: &dyn Fn(u64),
    ) -> Result<()> {
        let body = contents()[url];
        let existing = fs::metadata(output).map_or(0, |metadata| metadata.len()) as usize;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(output)?;
        file.write_all(&body[existing.min(body.len())..])?;
        progress(body.len() as u64);
        Ok(())
    }

    fn install_fixture(models: &Path, fetch: Fetch<'_>) -> Result<PathBuf> {
        install(
            models,
            &BUNDLE,
            &AtomicBool::new(false),
            &AtomicU64::new(0),
            &AtomicU8::new(0),
            fetch,
        )
    }

    #[test]
    fn installs_every_file_then_publishes_with_a_receipt() {
        let models = directory();
        let downloaded = AtomicU64::new(0);
        let path = install(
            &models,
            &BUNDLE,
            &AtomicBool::new(false),
            &downloaded,
            &AtomicU8::new(0),
            &resume_fetch,
        )
        .unwrap();
        assert_eq!(path, models.join("fixture-bundle-rev"));
        assert_eq!(downloaded.load(Ordering::Relaxed), 13);
        assert_eq!(BUNDLE.bytes(), 13);
        assert!(is_complete(&models, &BUNDLE));
        assert!(is_verified(&models, &BUNDLE));
        assert_eq!(
            fs::read(path.join("model.mlpackage/Data/weights.bin")).unwrap(),
            b"hello"
        );
        assert!(!models.join("fixture-bundle-rev.partial").exists());

        // A second install is a receipt check, never another download.
        install_fixture(&models, &|_, _, _, _| bail!("must not download")).unwrap();
        fs::remove_dir_all(models).unwrap();
    }

    #[test]
    fn an_interrupted_download_is_never_published_and_resumes() {
        let models = directory();
        let calls = RefCell::new(0);
        let interrupted = |url: &str, output: &Path, _: &AtomicBool, _: &dyn Fn(u64)| {
            *calls.borrow_mut() += 1;
            if url.ends_with("decoder.bin") {
                fs::write(output, b"wor")?;
                bail!("connection reset");
            }
            resume_fetch(url, output, &AtomicBool::new(false), &|_| {})
        };
        assert!(install_fixture(&models, &interrupted).is_err());
        assert!(!is_complete(&models, &BUNDLE));
        assert!(!is_verified(&models, &BUNDLE));
        assert!(!models.join("fixture-bundle-rev").exists());

        let fetched = RefCell::new(Vec::new());
        let resumed = |url: &str, output: &Path, canceled: &AtomicBool, progress: &dyn Fn(u64)| {
            let name = url.rsplit('/').next().unwrap().to_string();
            if name == "decoder.bin" {
                // curl resumes after the bytes the interrupted run kept.
                assert_eq!(fs::read(output).unwrap(), b"wor");
            }
            fetched.borrow_mut().push(name);
            resume_fetch(url, output, canceled, progress)
        };
        install_fixture(&models, &resumed).unwrap();
        // The verified weights from the first attempt were kept.
        assert_eq!(*fetched.borrow(), ["decoder.bin", "manifest.json"]);
        assert!(is_verified(&models, &BUNDLE));
        fs::remove_dir_all(models).unwrap();
    }

    #[test]
    fn a_corrupt_file_fails_without_publishing_and_is_redownloaded() {
        let models = directory();
        let corrupt = |url: &str, output: &Path, canceled: &AtomicBool, progress: &dyn Fn(u64)| {
            if url.ends_with("manifest.json") {
                fs::write(output, b"{]")?;
                return Ok(());
            }
            resume_fetch(url, output, canceled, progress)
        };
        let error = install_fixture(&models, &corrupt).unwrap_err().to_string();
        assert!(
            error.contains("checksum mismatch for manifest.json"),
            "{error}"
        );
        assert!(!models.join("fixture-bundle-rev").exists());
        assert!(
            !models
                .join("fixture-bundle-rev.partial/manifest.json")
                .exists()
        );
        install_fixture(&models, &resume_fetch).unwrap();
        assert!(is_verified(&models, &BUNDLE));
        fs::remove_dir_all(models).unwrap();
    }

    #[test]
    fn an_oversized_partial_is_discarded_before_resuming() {
        let models = directory();
        let staging = models.join("fixture-bundle-rev.partial");
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("decoder.bin"), b"world!!!").unwrap();
        install_fixture(&models, &resume_fetch).unwrap();
        assert!(is_verified(&models, &BUNDLE));
        fs::remove_dir_all(models).unwrap();
    }

    #[test]
    fn cancellation_stops_before_publication() {
        let models = directory();
        let canceled = AtomicBool::new(false);
        let cancel_midway =
            |url: &str, output: &Path, flag: &AtomicBool, progress: &dyn Fn(u64)| {
                resume_fetch(url, output, flag, progress)?;
                canceled.store(true, Ordering::Relaxed);
                Ok(())
            };
        let error = install(
            &models,
            &BUNDLE,
            &canceled,
            &AtomicU64::new(0),
            &AtomicU8::new(0),
            &cancel_midway,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("canceled"), "{error}");
        assert!(!models.join("fixture-bundle-rev").exists());
        fs::remove_dir_all(models).unwrap();
    }

    #[test]
    fn tampering_invalidates_the_receipt_and_is_repaired_by_replacement() {
        let models = directory();
        let path = install_fixture(&models, &resume_fetch).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        fs::write(path.join("decoder.bin"), b"WORLD!").unwrap();
        assert!(is_complete(&models, &BUNDLE));
        assert!(!is_verified(&models, &BUNDLE));
        assert!(verify_installed(&models, &BUNDLE, &AtomicBool::new(false)).is_err());
        // The installed-only check never repairs or deletes the bundle.
        assert_eq!(fs::read(path.join("decoder.bin")).unwrap(), b"WORLD!");

        install_fixture(&models, &resume_fetch).unwrap();
        assert!(is_verified(&models, &BUNDLE));
        assert_eq!(fs::read(path.join("decoder.bin")).unwrap(), b"world!");
        assert!(!models.join("fixture-bundle-rev.replaced").exists());
        fs::remove_dir_all(models).unwrap();
    }

    #[test]
    fn a_missing_receipt_is_restored_by_verification() {
        let models = directory();
        install_fixture(&models, &resume_fetch).unwrap();
        fs::remove_file(models.join("fixture-bundle-rev.verified.json")).unwrap();
        assert!(!is_verified(&models, &BUNDLE));
        verify_installed(&models, &BUNDLE, &AtomicBool::new(false)).unwrap();
        assert!(is_verified(&models, &BUNDLE));
        fs::remove_dir_all(models).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn symbolic_links_inside_a_bundle_are_rejected() {
        let models = directory();
        let path = install_fixture(&models, &resume_fetch).unwrap();
        let outside = models.join("outside.bin");
        fs::write(&outside, b"world!").unwrap();
        fs::remove_file(path.join("decoder.bin")).unwrap();
        std::os::unix::fs::symlink(&outside, path.join("decoder.bin")).unwrap();
        assert!(!is_complete(&models, &BUNDLE));
        assert!(!is_verified(&models, &BUNDLE));
        assert!(verify_installed(&models, &BUNDLE, &AtomicBool::new(false)).is_err());
        fs::remove_dir_all(models).unwrap();
    }

    #[test]
    fn removal_deletes_bundle_staging_and_receipt() {
        let models = directory();
        let path = install_fixture(&models, &resume_fetch).unwrap();
        fs::create_dir_all(path.join("Compiled.mlmodelc")).unwrap();
        fs::create_dir_all(models.join("fixture-bundle-rev.partial")).unwrap();
        remove(&models, &BUNDLE).unwrap();
        for leftover in [
            "fixture-bundle-rev",
            "fixture-bundle-rev.partial",
            "fixture-bundle-rev.verified.json",
        ] {
            assert!(!models.join(leftover).exists(), "{leftover}");
        }
        remove(&models, &BUNDLE).unwrap();
        fs::remove_dir_all(models).unwrap();
    }

    #[test]
    fn unsafe_specifications_are_rejected() {
        for path in ["../escape", "/absolute", "a/../b", "a\\b", ""] {
            assert!(safe_relative(path).is_err(), "{path}");
        }
        const BAD: ModelBundle = ModelBundle {
            directory: "..",
            repository: "fixture",
            revision: "rev",
            files: FILES,
        };
        assert!(validate_spec(&BAD).is_err());
        assert!(validate_spec(&BUNDLE).is_ok());
    }
}
