# set environment
$scriptDirectory = "E:\ImageClassificationRefundDep"
cd $scriptDirectory
.\venv\Scripts\activate.bat

# Logging
$logPath = "$scriptDirectory\output_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
Start-Transcript -Path $logPath -Append

# Settings for scheduled task
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Register inference task
$inferenceScriptPath = "E:\ImageClassificationRefundDep\client.py --inference"
$inferenceAction = New-ScheduledTaskAction -Execute "$scriptDirectory\venv\Scripts\python.exe" -Argument $inferenceScriptPath -WorkingDirectory $scriptDirectory
$inferenceTrigger = New-ScheduledTaskTrigger -Daily -At 20:00
Register-ScheduledTask -TaskName "m2p_test_timer" -Action $inferenceAction -Trigger $inferenceTrigger -Settings $settings -RunLevel Highest -Force

# Register analytics task
$analyticsScriptPath = "E:\ImageClassificationRefundDep\client.py --analytics"
$analyticsAction = New-ScheduledTaskAction -Execute "$scriptDirectory\venv\Scripts\python.exe" -Argument $analyticsScriptPath -WorkingDirectory $scriptDirectory
$analyticsTrigger = New-ScheduledTaskTrigger -Daily -At 23:59
Register-ScheduledTask -TaskName "m2p_test_timer" -Action $analyticsAction -Trigger $analyticsTrigger -Settings $settings -RunLevel Highest -Force

# logging end
Stop-Transcript
