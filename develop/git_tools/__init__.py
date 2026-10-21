"""Accès en lecture seule à Git (fichiers suivis, diff, log) via le binaire ``git``."""
from develop.git_tools.repo import (  # noqa: F401
    ChangedFile,
    CommitInfo,
    GitError,
    GitRepo,
    find_git_root,
    git_available,
)
