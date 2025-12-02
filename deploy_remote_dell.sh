#!/bin/bash
# macOS에서 Linux 서버로 프로젝트 배포 스크립트
# 원본에서 삭제된 파일도 대상에서 삭제합니다 (--delete 옵션)

# 스크립트가 위치한 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 배포 설정
SOURCE_DIR="/Users/deois/workspace/python/project_rl_v3"
REMOTE_USER="hdel"
REMOTE_HOST="www.hdel.io"
REMOTE_PATH="/home/hdel/workspace_pycharm/"

# 배포 전 확인 메시지
echo "🚀 프로젝트 배포를 시작합니다..."
echo "📍 원본: $SOURCE_DIR"
echo "📍 대상: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
echo ""
echo "⚠️  주의: --delete 옵션이 활성화되어 있어 원본에서 삭제된 파일이 대상에서도 삭제됩니다."
echo ""
read -p "계속하시겠습니까? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 배포가 취소되었습니다."
    exit 1
fi

# rsync 실행
# -a: 아카이브 모드 (권한, 타임스탬프 등 보존)
# -v: 상세 출력
# -z: 압축 전송
# --delete: 원본에 없는 파일을 대상에서 삭제
# --exclude: 제외할 디렉토리/파일
echo "📦 동기화 중..."
rsync -avz --delete \
    -e "ssh -p 45122" \
    --exclude '.venv' \
    --exclude '.mypy_cache' \
    --exclude '.pytest_cache' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.DS_Store' \
    --exclude 'logs' \
    --exclude 'archive' \
    "$SOURCE_DIR" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

# rsync 실행 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 배포가 성공적으로 완료되었습니다!"
else
    echo ""
    echo "❌ 배포 중 오류가 발생했습니다."
    exit 1
fi

