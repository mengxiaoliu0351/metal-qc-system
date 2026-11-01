param([string]$Server = '8.210.4.196', [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$accessDirectory = Join-Path $projectDirectory '.preview\aliyun-deploy-20261025\access'
$taskKeyPath = Join-Path $accessDirectory 'id_ed25519'
$taskHostsPath = Join-Path $accessDirectory 'known_hosts'
if (!(Test-Path -LiteralPath $taskKeyPath)) { throw 'Deployment key was not found on this computer.' }
$existingListener = Get-NetTCPConnection -LocalPort 18000 -State Listen -ErrorAction SilentlyContinue
if (!$existingListener) {
    $arguments = @('-N', '-i', '.preview/aliyun-deploy-20261025/access/id_ed25519', '-o', 'KexAlgorithms=curve25519-sha256',
        '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=.preview/aliyun-deploy-20261025/access/known_hosts',
        '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=30',
        '-o', 'ServerAliveCountMax=3', '-L', '127.0.0.1:18000:127.0.0.1:18000', "root@$Server")
    $process = Start-Process -FilePath 'ssh.exe' -ArgumentList $arguments -WorkingDirectory $projectDirectory -WindowStyle Hidden -PassThru `
        -RedirectStandardError (Join-Path $accessDirectory 'tunnel-error.log') -RedirectStandardOutput (Join-Path $accessDirectory 'tunnel-out.log')
    Set-Content -LiteralPath (Join-Path $accessDirectory 'admin-tunnel.pid') -Value $process.Id
    Start-Sleep -Seconds 2
    if ($process.HasExited) { throw 'SSH forwarding did not start; check the server connection.' }
}
if (!$NoBrowser) { Start-Process 'http://127.0.0.1:18000' }
Write-Host 'Private management entry: http://127.0.0.1:18000'
