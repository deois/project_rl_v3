#!/bin/bash
# 운영 모드로 Dash 앱 실행 스크립트
# .venv가 존재하면 자동으로 활성화하여 실행합니다.
# Linux와 macOS 모두에서 호환됩니다.

# 스크립트가 위치한 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 포트 8050을 사용하는 프로세스 확인 및 종료
PORT=8050

# lsof 또는 ss/netstat를 사용하여 포트를 사용하는 프로세스 찾기
find_port_process() {
    local port=$1
    local pid=""
    
    # macOS와 일부 Linux에서 lsof 사용
    if command -v lsof >/dev/null 2>&1; then
        pid=$(lsof -ti:$port 2>/dev/null)
    # Linux에서 ss 사용 (lsof가 없을 경우)
    elif command -v ss >/dev/null 2>&1; then
        pid=$(ss -ltnp 2>/dev/null | grep ":$port " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)
    # netstat 사용 (ss도 없을 경우)
    elif command -v netstat >/dev/null 2>&1; then
        pid=$(netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1 | head -1)
    fi
    
    echo "$pid"
}

PID=$(find_port_process $PORT)

if [ ! -z "$PID" ]; then
    echo "[WARNING] 포트 ${PORT}를 사용하는 프로세스(PID: ${PID})를 종료합니다."
    kill $PID 2>/dev/null
    sleep 1
    # 프로세스가 여전히 실행 중인지 확인
    PID_CHECK=$(find_port_process $PORT)
    if [ ! -z "$PID_CHECK" ]; then
        echo "[WARNING] 프로세스가 종료되지 않아 강제 종료합니다."
        kill -9 $PID 2>/dev/null
        sleep 1
    fi
    echo "[OK] 포트 ${PORT}가 해제되었습니다."
fi

# .venv 경로 설정
VENV_PATH="$SCRIPT_DIR/.venv"

# .venv가 존재하는지 확인
if [ -d "$VENV_PATH" ]; then
    # .venv의 Python 인터프리터 경로
    VENV_PYTHON="$VENV_PATH/bin/python"
    
    # .venv의 Python 인터프리터가 존재하는지 확인
    if [ -f "$VENV_PYTHON" ]; then
        echo "[OK] .venv를 찾았습니다. 가상 환경의 Python을 사용합니다."
        echo "[INFO] Python 경로: ${VENV_PYTHON}"
        # .venv의 Python을 사용하여 스크립트 실행
        exec "$VENV_PYTHON" "$SCRIPT_DIR/run_prod.py" "$@"
    else
        echo "[WARNING] .venv 디렉토리는 존재하지만 Python 인터프리터를 찾을 수 없습니다."
        echo "[INFO] 시스템 Python을 사용합니다."
        exec python3 "$SCRIPT_DIR/run_prod.py" "$@"
    fi
else
    echo "[INFO] .venv를 찾을 수 없습니다. 시스템 Python을 사용합니다."
    exec python3 "$SCRIPT_DIR/run_prod.py" "$@"
fi

