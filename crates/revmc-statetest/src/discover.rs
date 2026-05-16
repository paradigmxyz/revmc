use std::path::{Path, PathBuf};
use walkdir::{DirEntry, WalkDir};

/// Find all JSON test files under `paths`.
pub fn find_json_tests(
    paths: &[PathBuf],
    should_descend: fn(&Path) -> bool,
) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for path in paths {
        if path.is_file() {
            if is_json_test(path) {
                files.push(path.clone());
            }
            continue;
        }
        if !path.exists() {
            return Err(format!(
                "Path: {}\nName: Path validation\nError: path does not exist",
                path.display()
            ));
        }
        for entry in WalkDir::new(path)
            .follow_links(true)
            .into_iter()
            .filter_entry(|entry| should_walk_entry(entry, should_descend))
        {
            let entry = entry.map_err(|err| {
                format!("Path: {}\nName: Path validation\nError: walk error: {err}", path.display())
            })?;
            if entry.file_type().is_file() && is_json_test(entry.path()) {
                files.push(entry.path().to_path_buf());
            }
        }
    }
    files.sort_unstable();
    if files.is_empty() {
        return Err(
            "Path: \nName: Path validation\nError: no JSON test files found in path".to_string()
        );
    }
    Ok(files)
}

fn should_walk_entry(entry: &DirEntry, should_descend: fn(&Path) -> bool) -> bool {
    entry.depth() == 0 || should_descend(entry.path())
}

fn is_json_test(path: &Path) -> bool {
    path.file_name().is_none_or(|name| name != "index.json")
        && path.extension().is_some_and(|ext| ext == "json")
}
