"""
모델 관련 유틸리티 함수들
모델 파일 해시 계산, 검증 등
"""

import os
import hashlib
from typing import List


def calculate_model_hash(model_dir: str) -> str:
    """모델 파일들의 MD5 해시 계산"""
    try:
        hash_md5 = hashlib.md5()

        # Actor와 Critic 모델 파일들 확인
        model_files: List[str] = [
            os.path.join(model_dir, "actor.pth"),
            os.path.join(model_dir, "critic.pth"),
            os.path.join(model_dir, "actor_target.pth"),
            os.path.join(model_dir, "critic_target.pth")
        ]

        # 존재하는 모델 파일들만 해시 계산
        existing_files: List[str] = []
        for file_path in model_files:
            if os.path.exists(file_path):
                existing_files.append(file_path)

        if not existing_files:
            return "no_model_files"

        # 파일들을 정렬하여 일관된 해시 생성
        existing_files.sort()

        for file_path in existing_files:
            with open(file_path, "rb") as f:
                # 파일을 청크 단위로 읽어서 메모리 효율성 증대
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)

        return hash_md5.hexdigest()

    except Exception as e:
        return f"hash_error_{str(e)[:10]}"


def validate_model_files(model_dir: str) -> bool:
    """모델 파일들의 존재 여부 검증"""
    required_files = [
        "actor.pth",
        "critic.pth",
        "actor_target.pth",
        "critic_target.pth"
    ]

    for file_name in required_files:
        file_path = os.path.join(model_dir, file_name)
        if not os.path.exists(file_path):
            return False

    return True


def get_model_file_sizes(model_dir: str) -> dict:
    """모델 파일들의 크기 정보 반환"""
    file_sizes = {}
    model_files = [
        "actor.pth",
        "critic.pth",
        "actor_target.pth",
        "critic_target.pth"
    ]

    for file_name in model_files:
        file_path = os.path.join(model_dir, file_name)
        if os.path.exists(file_path):
            file_sizes[file_name] = os.path.getsize(file_path)
        else:
            file_sizes[file_name] = 0

    return file_sizes
