# Files Excluded from the Repository

The original working folder contained many files that should not be included in a public GitHub repository. This release keeps only the core files required to understand the data format, reproduce the main experiments, regenerate paper figures, and test Arduino deployment.

Excluded categories:

```text
_unused_datasets_backup_20260717/   Old dataset backups
.codex_tmp/                         Temporary and automatically generated files
*.npy                               Local training caches
__pycache__/                        Python bytecode caches
private thesis/front-matter PDFs    Approval forms, recommendation forms, and authorization documents
teacher paper draft files           Teacher paper drafts and internal discussion files
large generated output folders      Large intermediate outputs and duplicated experiment folders
```

The repository is intended to remain compact, reproducible, and safe to share publicly.
