#!/bin/sh
# install.sh — installe Develop sur macOS / Linux, sans Docker ni sudo.
#
#   sh install.sh [--desktop] [--python /chemin/python3] [--from CHEMIN] [--bin-dir DOSSIER]
#
# Ce que fait le script :
#   1. cherche un Python 3.9 ou plus récent (et explique comment l'installer sinon) ;
#   2. crée l'environnement ~/.develop/venv (DEVELOP_HOME pour changer de dossier) ;
#   3. y installe Develop depuis ce dépôt, ou depuis --from (dossier, archive .tar.gz/.zip, wheel) ;
#   4. crée la commande « develop » dans ~/.local/bin (DEVELOP_BIN_DIR ou --bin-dir pour changer).
#
# Relançable sans risque : l'environnement existant est réutilisé et le paquet mis à jour.
# Seul pip utilise le réseau. Aucune commande n'est lancée avec sudo.
set -eu

DEVELOP_HOME="${DEVELOP_HOME:-$HOME/.develop}"
VENV="$DEVELOP_HOME/venv"
BIN_DIR="${DEVELOP_BIN_DIR:-$HOME/.local/bin}"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SOURCE="$SCRIPT_DIR"
PYTHON="${DEVELOP_PYTHON:-}"
EXTRAS=""
SHIM_MARK="# Généré par install.sh de Develop"

info() { printf '%s\n' "$*"; }
die() { printf 'Erreur : %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Installe Develop (macOS / Linux) dans ~/.develop/venv et crée la commande « develop ».

Usage : sh install.sh [options]
  --desktop            installe aussi pywebview (fenêtre native « develop app »)
  --python CHEMIN      Python à utiliser (sinon le premier Python >= 3.9 trouvé dans le PATH)
  --from CHEMIN        source à installer : dossier du dépôt (défaut : ce dossier),
                       archive .tar.gz/.zip ou wheel .whl
  --bin-dir DOSSIER    dossier de la commande « develop » (défaut : ~/.local/bin)
  -h, --help           cette aide

Variables : DEVELOP_HOME (défaut ~/.develop), DEVELOP_PYTHON, DEVELOP_BIN_DIR.
EOF
}

need_value() { [ $# -ge 2 ] || die "l'option $1 attend une valeur."; }

while [ $# -gt 0 ]; do
  case "$1" in
    --desktop) EXTRAS="[desktop]" ;;
    --python) need_value "$@"; shift; PYTHON="$1" ;;
    --python=*) PYTHON="${1#--python=}" ;;
    --from) need_value "$@"; shift; SOURCE="$1" ;;
    --from=*) SOURCE="${1#--from=}" ;;
    --bin-dir) need_value "$@"; shift; BIN_DIR="$1" ;;
    --bin-dir=*) BIN_DIR="${1#--bin-dir=}" ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Option inconnue : %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# ----------------------------------------------------------------------------- 1. Python >= 3.9

python_version_ok() {  # $1 = commande Python ; vrai si elle existe et vaut 3.9 ou plus
  command -v "$1" >/dev/null 2>&1 || return 1
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1
}

python_has_venv() {  # Debian/Ubuntu livrent python3 sans le module venv (paquet python3-venv)
  "$1" -c 'import venv, ensurepip' >/dev/null 2>&1
}

suggest_python() {
  case "$(uname -s)" in
    Darwin)
      info "Installez Python 3.11 ou plus récent :"
      info "  - avec Homebrew : brew install python"
      info "  - ou depuis https://www.python.org/downloads/macos/" ;;
    *)
      info "Installez Python 3.9 ou plus récent avec votre gestionnaire de paquets, par exemple :"
      info "  - Debian/Ubuntu : sudo apt install python3 python3-venv"
      info "  - Fedora        : sudo dnf install python3"
      info "  - Arch          : sudo pacman -S python"
      info "  - ou depuis https://www.python.org/downloads/" ;;
  esac
  info "puis relancez : sh install.sh"
}

if [ -n "$PYTHON" ]; then
  python_version_ok "$PYTHON" || die "« $PYTHON » n'est pas un Python 3.9 ou plus récent (ou n'est pas exécutable)."
else
  for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
    if python_version_ok "$candidate"; then
      PYTHON="$candidate"
      break
    fi
  done
  if [ -z "$PYTHON" ]; then
    info "Aucun Python 3.9 ou plus récent trouvé dans le PATH." >&2
    suggest_python >&2
    exit 1
  fi
fi
if ! python_has_venv "$PYTHON"; then
  info "« $PYTHON » n'a pas le module venv/ensurepip." >&2
  case "$(uname -s)" in
    Linux) info "Sur Debian/Ubuntu : sudo apt install python3-venv (puis relancez ce script)." >&2 ;;
    *) suggest_python >&2 ;;
  esac
  exit 1
fi
PY_VERSION=$("$PYTHON" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
info "Python : $PYTHON ($PY_VERSION)"

# ----------------------------------------------------------------------------- 2. environnement

venv_usable() {
  [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1
}

if venv_usable; then
  info "Environnement : $VENV (existant, réutilisé)"
elif [ -e "$VENV" ] && [ ! -f "$VENV/pyvenv.cfg" ]; then
  die "$VENV existe mais n'est pas un environnement Python : déplacez-le ou changez DEVELOP_HOME."
else
  if [ -e "$VENV" ]; then
    info "Environnement : $VENV (inutilisable, recréé)"
    rm -rf "$VENV"
  else
    info "Environnement : $VENV (création)"
  fi
  mkdir -p "$DEVELOP_HOME"
  "$PYTHON" -m venv "$VENV" || die "la création de l'environnement a échoué."
fi
VENV_PY="$VENV/bin/python"

# Un pip trop ancien (Python système de macOS : 21.2) copie tout le dépôt avant de construire et
# choisit mal certains wheels : on le met à jour une fois, en dessous de la version 22.
if ! "$VENV_PY" -c 'import pip; v = tuple(int(x) for x in pip.__version__.split(".")[:2]); raise SystemExit(0 if v >= (22, 0) else 1)' >/dev/null 2>&1; then
  info "Mise à jour de pip…"
  "$VENV_PY" -m pip install --disable-pip-version-check --quiet --upgrade pip \
    || info "(pip n'a pas pu être mis à jour, on continue avec la version présente)"
fi

# ----------------------------------------------------------------------------- 3. installation

if [ -d "$SOURCE" ]; then
  SOURCE=$(cd "$SOURCE" && pwd)
  [ -f "$SOURCE/pyproject.toml" ] || die "aucun pyproject.toml dans $SOURCE : lancez ce script depuis le dépôt Develop cloné, ou indiquez --from."
elif [ -f "$SOURCE" ]; then
  case "$SOURCE" in
    /*) ;;
    *) SOURCE="$(pwd)/$SOURCE" ;;
  esac
else
  die "source introuvable : $SOURCE"
fi
EXTRAS_NOTE=""
[ -n "$EXTRAS" ] && EXTRAS_NOTE=" (avec l'extra desktop)"
info "Installation de Develop depuis ${SOURCE}${EXTRAS_NOTE}…"
"$VENV_PY" -m pip install --disable-pip-version-check --quiet "$SOURCE$EXTRAS" \
  || die "pip n'a pas pu installer Develop (voir les messages ci-dessus)."
VERSION=$("$VENV_PY" -m develop --version 2>/dev/null) || die "Develop est installé mais « develop --version » échoue."
info "Installé : $VERSION"

# ----------------------------------------------------------------------------- 4. commande develop

SHIM="$BIN_DIR/develop"
if [ -e "$SHIM" ] && ! grep -q "$SHIM_MARK" "$SHIM" 2>/dev/null; then
  die "$SHIM existe déjà et n'a pas été créé par ce script : supprimez-le ou utilisez --bin-dir."
fi
mkdir -p "$BIN_DIR"
cat > "$SHIM" <<EOF
#!/bin/sh
$SHIM_MARK
# Lance Develop depuis son environnement. Relancer install.sh met à jour l'installation.
exec "$VENV_PY" -m develop "\$@"
EOF
chmod 755 "$SHIM"
info "Commande : $SHIM"

case ":$PATH:" in
  *":$BIN_DIR:"*) ON_PATH=1 ;;
  *) ON_PATH=0 ;;
esac

info ""
info "Develop est installé."
if [ "$ON_PATH" -eq 0 ]; then
  info "Le dossier $BIN_DIR n'est pas dans votre PATH. Ajoutez-le, puis ouvrez un nouveau terminal :"
  case "$(basename "${SHELL:-sh}")" in
    zsh)  info "  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.zshrc" ;;
    bash) if [ "$(uname -s)" = Darwin ]; then RC="~/.bash_profile"; else RC="~/.bashrc"; fi
          info "  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> $RC" ;;
    fish) info "  fish_add_path $BIN_DIR" ;;
    *)    info "  export PATH=\"$BIN_DIR:\$PATH\"   (dans le fichier de démarrage de votre shell)" ;;
  esac
  info "En attendant, la commande complète fonctionne : $SHIM"
fi
info ""
info "Prochaines étapes :"
info "  develop doctor     diagnostic de la machine et modèles recommandés"
info "  develop serve      interface dans le navigateur (http://127.0.0.1:8765)"
if [ -n "$EXTRAS" ]; then
  info "  develop app        interface dans une fenêtre native"
  if [ "$(uname -s)" = Linux ]; then
    info "                     (sous Linux, pywebview a besoin de GTK/WebKit : python3-gi gir1.2-webkit2-4.1, ou de Qt)"
  fi
else
  info "  develop app        fenêtre native : relancez avec --desktop pour l'activer"
fi
info "  sh install.sh      relancer pour mettre à jour"
