param(
    [Parameter(Position = 0)]
    [string]$Message = "",
    [switch]$SkipDeploy
)

$scriptPath = Join-Path $PSScriptRoot "sync_code.py"
$arguments = @($scriptPath)

if ($Message) {
    $arguments += $Message
}

if ($SkipDeploy) {
    $arguments += "--skip-deploy"
}

python $arguments
