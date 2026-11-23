$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path 'vendor_liblinear/liblinear')) {
    python -m pip install --no-deps --target vendor_liblinear liblinear-official==2.50.0
    if ($LASTEXITCODE -ne 0) { throw 'LIBLINEAR installation failed' }
}
function Invoke-PilotPython {
    param([string[]]$Arguments)
    & python -u @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Pilot stage failed: $Arguments" }
}
Invoke-PilotPython -Arguments @('boundary_cosmetics.py')
Invoke-PilotPython -Arguments @('validate.py')
foreach ($dataset in @('Coveo','Cosmetics')) {
    Invoke-PilotPython -Arguments @('pilot.py','prepare',$dataset)
}
Invoke-PilotPython -Arguments @('audit_price_cleaning.py')
Invoke-PilotPython -Arguments @('audit_coveo_search_closure.py')
foreach ($dataset in @('Coveo','Cosmetics')) {
    Invoke-PilotPython -Arguments @('mine.py',$dataset)
}
Invoke-PilotPython -Arguments @('audit_features.py')
Invoke-PilotPython -Arguments @('audit_edge_margins.py')
foreach ($dataset in @('Coveo','Cosmetics')) {
    Invoke-PilotPython -Arguments @('fit.py','fit',$dataset)
}
foreach ($dataset in @('Coveo','Cosmetics')) {
    Invoke-PilotPython -Arguments @('fit.py','evaluate',$dataset)
}
Invoke-PilotPython -Arguments @('report.py')
Invoke-PilotPython -Arguments @('final_validate.py')
