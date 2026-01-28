@echo off
REM SolarEye System - Development Server Startup Script

echo.
echo ==========================================
echo   SolarEye System - Development Startup
echo ==========================================
echo.

REM Start API Server in one window
echo [1/2] Starting Flask API Server on port 5000...
start "SolarEye API Server" python api_server.py

REM Wait a bit for API to start
timeout /t 3

REM Start Web Server in another window
echo [2/2] Starting Web Server on port 8000...
cd Web_Implementation
start "SolarEye Web Server" python -m http.server 8000

cd ..

echo.
echo ==========================================
echo   Servers Started Successfully!
echo ==========================================
echo.
echo API Server:  http://127.0.0.1:5000
echo Web Server:  http://127.0.0.1:8000
echo.
echo Opening browser...
timeout /t 2
start http://127.0.0.1:8000/index.html

echo.
echo Done! Both servers are running.
echo Close either window to stop the respective server.
echo.
pause
