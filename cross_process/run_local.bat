@echo off
REM NOTE: sur certaines machines Windows, gloo ne parvient pas a ouvrir les liaisons entre processus (erreur 10049).
REM Dans ce cas, lancer plutot sous WSL/Ubuntu : bash cross_process/run_local.sh iid 5 CP_iid
REM Launch 1 server + 2 client processes on Windows (3 separate console windows).
REM Usage: cross_process\run_local.bat [partition] [rounds] [tag]
set PARTITION=%1
if "%PARTITION%"=="" set PARTITION=iid
set ROUNDS=%2
if "%ROUNDS%"=="" set ROUNDS=20
set TAG=%3
if "%TAG%"=="" set TAG=CP_%PARTITION%
set WORLD=3
cd /d "%~dp0\.."
if not exist results mkdir results
start "FL server" cmd /k python cross_process\server.py --world_size %WORLD% --rounds %ROUNDS% --partition %PARTITION% --tag %TAG%
timeout /t 3 /nobreak >nul
start "FL client 1" cmd /k python cross_process\client.py --world_size %WORLD% --rank 1 --partition %PARTITION%
start "FL client 2" cmd /k python cross_process\client.py --world_size %WORLD% --rank 2 --partition %PARTITION%
echo Launched server + 2 clients. Results will be in results\%TAG%.json
