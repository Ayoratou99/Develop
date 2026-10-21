<#
.SYNOPSIS
Installe Develop sous Windows, sans Docker ni droits administrateur.

.DESCRIPTION
    powershell -ExecutionPolicy Bypass -File install.ps1 [-Desktop] [-Python chemin] [-From chemin] [-BinDir dossier]

Ce que fait le script :
  1. cherche un Python 3.9 ou plus récent (py -3, python3, python) et explique comment l'installer sinon ;
  2. crée l'environnement %USERPROFILE%\.develop\venv (DEVELOP_HOME pour changer de dossier) ;
  3. y installe Develop depuis ce dépôt, ou depuis -From (dossier, archive .tar.gz/.zip, wheel) ;
  4. crée la commande « develop » dans %USERPROFILE%\.develop\bin et ajoute ce dossier au PATH utilisateur.

Relançable sans risque : l'environnement existant est réutilisé et le paquet mis à jour.
Seul pip utilise le réseau. Rien n'est écrit hors du profil utilisateur.

.PARAMETER Desktop
Installe aussi pywebview (fenêtre native « develop app », via WebView2 fourni avec Windows 10/11).
.PARAMETER Python
Python à utiliser (sinon le premier Python >= 3.9 trouvé).
.PARAMETER From
Source à installer : dossier du dépôt (défaut : le dossier du script), archive ou wheel.
.PARAMETER BinDir
Dossier de la commande « develop » (défaut : %USERPROFILE%\.develop\bin, ou DEVELOP_BIN_DIR).
#>
[CmdletBinding()]
param(
    [switch]$Desktop,
    [string]$Python = $env:DEVELOP_PYTHON,
    [string]$From = $PSScriptRoot,
    [string]$BinDir = $env:DEVELOP_BIN_DIR
)

$ErrorActionPreference = "Stop"
$ShimMark = "rem Genere par install.ps1 de Develop"

$DevelopHome = if ($env:DEVELOP_HOME) { $env:DEVELOP_HOME } else { Join-Path $env:USERPROFILE ".develop" }
$Venv = Join-Path $DevelopHome "venv"
if (-not $BinDir) { $BinDir = Join-Path $DevelopHome "bin" }
if (-not $From) { $From = (Get-Location).Path }

function Fail([string]$Message) {
    Write-Host "Erreur : $Message" -ForegroundColor Red
    exit 1
}

# Lance une commande native et renvoie son code de retour. Dans Windows PowerShell 5.1, une sortie
# d'erreur redirigée devient une exception quand $ErrorActionPreference vaut Stop : on l'évite ici.
function Invoke-Native([string]$Exe, [string[]]$Arguments, [switch]$Quiet) {
    $ErrorActionPreference = "Continue"
    if ($Quiet) {
        $null = & $Exe @Arguments 2>&1
    } else {
        & $Exe @Arguments
    }
    return $LASTEXITCODE
}

# Une commande Python peut être « py -3 » : on sépare l'exécutable de ses arguments.
function Split-Command([string]$Command) {
    $parts = $Command.Trim() -split "\s+"
    $extra = if ($parts.Length -gt 1) { $parts[1..($parts.Length - 1)] } else { @() }
    return @{ Exe = $parts[0]; Args = @($extra) }
}

function Test-PythonOk([string]$Command) {
    $cmd = Split-Command $Command
    if (-not (Get-Command $cmd.Exe -ErrorAction SilentlyContinue)) { return $false }
    # L'alias « python » du Microsoft Store ouvre le magasin et échoue : le code de retour le révèle.
    $check = "import sys, venv, ensurepip; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)"
    $code = Invoke-Native $cmd.Exe (@($cmd.Args) + @("-c", $check)) -Quiet
    return ($code -eq 0)
}

# ----------------------------------------------------------------------------- 1. Python >= 3.9

if ($Python) {
    if (-not (Test-PythonOk $Python)) { Fail "« $Python » n'est pas un Python 3.9 ou plus récent." }
} else {
    foreach ($candidate in @("py -3", "python3", "python")) {
        if (Test-PythonOk $candidate) { $Python = $candidate; break }
    }
    if (-not $Python) {
        Write-Host "Aucun Python 3.9 ou plus récent trouvé." -ForegroundColor Yellow
        Write-Host "Installez Python 3.11 ou plus récent :"
        Write-Host "  - winget install Python.Python.3.12"
        Write-Host "  - ou depuis https://www.python.org/downloads/windows/ (cochez « Add python.exe to PATH »)"
        Write-Host "puis relancez : powershell -ExecutionPolicy Bypass -File install.ps1"
        exit 1
    }
}
$PyCmd = Split-Command $Python
$versionArgs = @($PyCmd.Args) + @("-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])")
$PyVersion = [string](& $PyCmd.Exe @versionArgs)
Write-Host "Python : $Python ($($PyVersion.Trim()))"

# ----------------------------------------------------------------------------- 2. environnement

$VenvPy = Join-Path $Venv "Scripts\python.exe"
function Test-VenvUsable {
    if (-not (Test-Path -LiteralPath $VenvPy)) { return $false }
    $code = Invoke-Native $VenvPy @("-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)") -Quiet
    return ($code -eq 0)
}

if (Test-VenvUsable) {
    Write-Host "Environnement : $Venv (existant, réutilisé)"
} elseif ((Test-Path -LiteralPath $Venv) -and -not (Test-Path -LiteralPath (Join-Path $Venv "pyvenv.cfg"))) {
    Fail "$Venv existe mais n'est pas un environnement Python : déplacez-le ou changez DEVELOP_HOME."
} else {
    if (Test-Path -LiteralPath $Venv) {
        Write-Host "Environnement : $Venv (inutilisable, recréé)"
        Remove-Item -LiteralPath $Venv -Recurse -Force
    } else {
        Write-Host "Environnement : $Venv (création)"
    }
    New-Item -ItemType Directory -Force -Path $DevelopHome | Out-Null
    $code = Invoke-Native $PyCmd.Exe (@($PyCmd.Args) + @("-m", "venv", $Venv))
    if ($code -ne 0) { Fail "la création de l'environnement a échoué." }
}

# Un pip trop ancien copie tout le dépôt avant de construire et choisit mal certains wheels :
# mise à jour une fois, en dessous de la version 22.
$pipCheck = "import pip; v = tuple(int(x) for x in pip.__version__.split('.')[:2]); raise SystemExit(0 if v >= (22, 0) else 1)"
if ((Invoke-Native $VenvPy @("-c", $pipCheck) -Quiet) -ne 0) {
    Write-Host "Mise à jour de pip..."
    $code = Invoke-Native $VenvPy @("-m", "pip", "install", "--disable-pip-version-check", "--quiet", "--upgrade", "pip")
    if ($code -ne 0) { Write-Host "(pip n'a pas pu être mis à jour, on continue avec la version présente)" }
}

# ----------------------------------------------------------------------------- 3. installation

if (Test-Path -LiteralPath $From -PathType Container) {
    $From = (Resolve-Path -LiteralPath $From).Path
    if (-not (Test-Path -LiteralPath (Join-Path $From "pyproject.toml"))) {
        Fail "aucun pyproject.toml dans $From : lancez ce script depuis le dépôt Develop cloné, ou indiquez -From."
    }
} elseif (Test-Path -LiteralPath $From -PathType Leaf) {
    $From = (Resolve-Path -LiteralPath $From).Path
} else {
    Fail "source introuvable : $From"
}
$Extras = if ($Desktop) { "[desktop]" } else { "" }
$suffix = if ($Desktop) { " (avec l'extra desktop)" } else { "" }
Write-Host "Installation de Develop depuis $From$suffix..."
$code = Invoke-Native $VenvPy @("-m", "pip", "install", "--disable-pip-version-check", "--quiet", "$From$Extras")
if ($code -ne 0) { Fail "pip n'a pas pu installer Develop (voir les messages ci-dessus)." }
$Version = [string](& $VenvPy -m develop --version)
if ($LASTEXITCODE -ne 0) { Fail "Develop est installé mais « develop --version » échoue." }
Write-Host "Installé : $($Version.Trim())"

# ----------------------------------------------------------------------------- 4. commande develop

$Shim = Join-Path $BinDir "develop.cmd"
if ((Test-Path -LiteralPath $Shim) -and -not ((Get-Content -LiteralPath $Shim -Raw) -like "*$ShimMark*")) {
    Fail "$Shim existe déjà et n'a pas été créé par ce script : supprimez-le ou utilisez -BinDir."
}
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
# Dans le profil par défaut, le chemin est écrit avec %USERPROFILE% : le .cmd reste en ASCII
# même si le nom d'utilisateur contient des accents.
$defaultHome = Join-Path $env:USERPROFILE ".develop"
$shimPython = if ($DevelopHome -eq $defaultHome) { '%USERPROFILE%\.develop\venv\Scripts\python.exe' } else { $VenvPy }
[string[]]$shimLines = @(
    "@echo off",
    $ShimMark,
    "rem Lance Develop depuis son environnement. Relancer install.ps1 met a jour l'installation.",
    "`"$shimPython`" -m develop %*"
)
[System.IO.File]::WriteAllLines($Shim, $shimLines, [System.Text.Encoding]::ASCII)
Write-Host "Commande : $Shim"

# PATH utilisateur (registre, type ExpandString conservé), puis notification aux autres programmes.
$regKey = "HKCU:\Environment"
$rawPath = [string](Get-Item -LiteralPath $regKey).GetValue("Path", "", "DoNotExpandEnvironmentNames")
$entries = @($rawPath -split ";" | Where-Object { $_ })
$expanded = @($entries | ForEach-Object { [Environment]::ExpandEnvironmentVariables($_).TrimEnd("\") })
if ($expanded -notcontains $BinDir.TrimEnd("\")) {
    $newPath = (@($entries) + $BinDir) -join ";"
    New-ItemProperty -Path $regKey -Name "Path" -Value $newPath -PropertyType ExpandString -Force | Out-Null
    try {
        if (-not ("Develop.Env" -as [type])) {
            Add-Type -Namespace Develop -Name Env -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("user32.dll", SetLastError = true, CharSet = System.Runtime.InteropServices.CharSet.Auto)]
public static extern System.IntPtr SendMessageTimeout(System.IntPtr hWnd, uint Msg, System.UIntPtr wParam, string lParam, uint fuFlags, uint uTimeout, out System.UIntPtr lpdwResult);
'@
        }
        $result = [System.UIntPtr]::Zero
        [Develop.Env]::SendMessageTimeout([System.IntPtr]0xffff, 0x001A, [System.UIntPtr]::Zero, "Environment", 2, 5000, [ref]$result) | Out-Null
    } catch {
        # Sans notification, le nouveau PATH est pris en compte à la prochaine session.
    }
    Write-Host "Ajouté au PATH utilisateur : $BinDir (ouvrez un nouveau terminal pour l'utiliser)"
}
if (($env:Path -split ";") -notcontains $BinDir) { $env:Path = "$env:Path;$BinDir" }

Write-Host ""
Write-Host "Develop est installé." -ForegroundColor Green
Write-Host "Prochaines étapes (dans un nouveau terminal) :"
Write-Host "  develop doctor     diagnostic de la machine et modèles recommandés"
Write-Host "  develop serve      interface dans le navigateur (http://127.0.0.1:8765)"
if ($Desktop) {
    Write-Host "  develop app        interface dans une fenêtre native"
} else {
    Write-Host "  develop app        fenêtre native : relancez avec -Desktop pour l'activer"
}
Write-Host "  install.ps1        relancer pour mettre à jour"
