"""
Dash 앱 유틸리티 함수들
모델 스캔, 설정 관리 등
"""

import os
import json
import time
import shutil
from typing import List, Dict, Any, Optional
from src.utils.logger import get_logger

# 로거 설정
logger = get_logger("dash_utils")


def get_model_metadata(model_path: str) -> Optional[Dict[str, Any]]:
    """모델의 메타데이터를 읽어옵니다."""
    try:
        # JSON 메타데이터 파일 확인
        metadata_file = os.path.join(model_path, "metadata_last.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                return {
                    "episode": metadata.get("episode", 0),
                    "date": time.strftime(
                        "%m-%d %H:%M", time.localtime(metadata.get("save_time", 0))
                    ),
                    "full_metadata": metadata,
                }

        # PyTorch 체크포인트에서 메타데이터 확인
        checkpoint_file = os.path.join(model_path, "checkpoint_last.pth")
        if os.path.exists(checkpoint_file):
            try:
                import torch

                # PyTorch 2.6 호환성을 위해 weights_only=False 명시적 설정
                checkpoint = torch.load(
                    checkpoint_file, map_location="cpu", weights_only=False
                )
                if (
                    "training_metadata" in checkpoint
                    and checkpoint["training_metadata"]
                ):
                    metadata = checkpoint["training_metadata"]
                    episode = checkpoint.get(
                        "episode", metadata.get("current_episode", 0)
                    )
                    save_time = checkpoint.get("save_time", time.time())
                    return {
                        "episode": episode,
                        "date": time.strftime("%m-%d %H:%M", time.localtime(save_time)),
                        "full_metadata": metadata,
                    }
            except Exception as e:
                logger.debug(f"PyTorch 체크포인트 읽기 실패: {e}")

        return None
    except Exception as e:
        logger.debug(f"메타데이터 읽기 실패 ({model_path}): {e}")
        return None


def load_model_training_config(model_path: str) -> Optional[Dict[str, Any]]:
    """모델의 학습 설정을 로드합니다."""
    metadata = get_model_metadata(model_path)
    if metadata and "full_metadata" in metadata:
        return metadata["full_metadata"].get("training_config", {})
    return None


def get_latest_episode_from_model(model_path: str) -> int:
    """모델 디렉토리에서 실제로 저장된 최신 에피소드 번호를 찾습니다."""
    try:
        # 먼저 메타데이터에서 확인
        metadata_info = get_model_metadata(model_path)
        if metadata_info and metadata_info.get("episode", 0) > 0:
            return metadata_info["episode"]

        # 메타데이터가 없거나 에피소드가 0인 경우, 체크포인트 파일들 직접 스캔
        if os.path.exists(model_path):
            checkpoint_files = []
            for file in os.listdir(model_path):
                if (
                    file.startswith("checkpoint_")
                    and file.endswith(".pth")
                    and "last" not in file
                ):
                    try:
                        # checkpoint_0001.pth 형식에서 숫자 추출
                        episode_str = file.replace("checkpoint_", "").replace(
                            ".pth", ""
                        )
                        episode_num = int(episode_str)
                        checkpoint_files.append(episode_num)
                    except ValueError:
                        continue

            if checkpoint_files:
                latest_episode = max(checkpoint_files)
                logger.info(
                    f"체크포인트 파일 스캔으로 최신 에피소드 발견: {latest_episode}"
                )
                return latest_episode

        logger.warning(f"모델 {model_path}에서 유효한 에피소드를 찾을 수 없습니다")
        return 0

    except Exception as e:
        logger.error(f"최신 에피소드 확인 중 오류 ({model_path}): {e}")
        return 0


def get_available_models() -> List[Dict[str, str]]:
    """백테스팅에 사용 가능한 모델 목록 반환 (필수 파일 체크 및 무효한 폴더 삭제)"""
    model_options = []
    deleted_folders = []

    try:
        # ./model 디렉토리 확인
        model_base_dir = "./model"
        if os.path.exists(model_base_dir):
            # rl_ddpg로 시작하는 디렉토리들 찾기
            for item in os.listdir(model_base_dir):
                item_path = os.path.join(model_base_dir, item)
                if os.path.isdir(item_path) and item.startswith("rl_ddpg"):

                    # 백테스팅에 필요한 필수 파일들 확인
                    checkpoint_file = os.path.join(item_path, "checkpoint_last.pth")
                    metadata_file = os.path.join(item_path, "metadata_last.json")

                    # 두 파일이 모두 존재하는 경우만 추가
                    if os.path.exists(checkpoint_file) and os.path.exists(
                        metadata_file
                    ):
                        # 메타데이터 확인
                        metadata_info = get_model_metadata(item_path)
                        if metadata_info:
                            label = f"📁 {item} (E{metadata_info['episode']}, {metadata_info['date']})"
                        else:
                            label = f"📁 {item}"

                        model_options.append({"label": label, "value": item_path})
                        logger.debug(f"✅ 백테스팅 가능한 모델 발견: {item_path}")
                    else:
                        # 필수 파일이 누락된 경우 폴더 삭제
                        missing_files = []
                        if not os.path.exists(checkpoint_file):
                            missing_files.append("checkpoint_last.pth")
                        if not os.path.exists(metadata_file):
                            missing_files.append("metadata_last.json")

                        # 기본 모델 디렉토리가 아닌 경우에만 삭제 (안전장치)
                        if item not in ["rl_ddpg", "rl_ddpg_latest"]:
                            try:
                                logger.info(
                                    f"🗑️ 필수 파일 누락으로 인한 모델 폴더 삭제: {item_path}"
                                )
                                logger.info(
                                    f"   누락된 파일: {', '.join(missing_files)}"
                                )
                                shutil.rmtree(item_path)
                                deleted_folders.append(item)
                                logger.info(f"✅ 폴더 삭제 완료: {item}")
                            except Exception as e:
                                logger.error(f"❌ 폴더 삭제 실패 ({item}): {str(e)}")
                        else:
                            logger.debug(
                                f"⚠️ 기본 모델 {item}에 필수 파일 누락: {', '.join(missing_files)} (삭제하지 않음)"
                            )

        # 기본 모델 경로들도 동일한 조건으로 확인 (삭제하지 않음)
        default_models = [
            {"label": "🎯 기본 DDPG 모델", "value": "./model/rl_ddpg"},
            {"label": "📊 최신 체크포인트", "value": "./model/rl_ddpg_latest"},
        ]

        # 중복 제거하면서 기본 모델들 추가 (필수 파일 체크)
        existing_values = {opt["value"] for opt in model_options}
        for default_model in default_models:
            if default_model["value"] not in existing_values:
                # 기본 모델도 필수 파일 확인
                checkpoint_file = os.path.join(
                    default_model["value"], "checkpoint_last.pth"
                )
                metadata_file = os.path.join(
                    default_model["value"], "metadata_last.json"
                )

                if os.path.exists(checkpoint_file) and os.path.exists(metadata_file):
                    metadata_info = get_model_metadata(default_model["value"])
                    if metadata_info:
                        default_model[
                            "label"
                        ] += f" (E{metadata_info['episode']}, {metadata_info['date']})"
                    model_options.insert(0, default_model)
                    logger.debug(f"✅ 기본 모델 추가: {default_model['value']}")

        # 삭제 결과 로깅
        if deleted_folders:
            logger.info(
                f"🧹 정리 완료: {len(deleted_folders)}개 무효한 모델 폴더 삭제됨 ({', '.join(deleted_folders)})"
            )

        logger.info(f"📊 백테스팅 가능한 모델 {len(model_options)}개 발견")

    except Exception as e:
        logger.error(f"모델 디렉토리 스캔 중 오류: {e}")
        model_options = []

    return (
        model_options
        if model_options
        else [{"label": "❌ 백테스팅 가능한 모델 없음 (필수 파일 누락)", "value": ""}]
    )


# 🎨 스타일 상수들
CARD_STYLE = {
    "margin": "8px",
    "border-radius": "12px",
    "box-shadow": "0 2px 8px rgba(0, 0, 0, 0.1)",
    "border": "1px solid rgba(0, 0, 0, 0.05)",
}

METRIC_CARD_STYLE = {
    **CARD_STYLE,
    "text-align": "center",
    "height": "120px",
    "padding": "8px",
}

CUSTOM_CSS = {
    "font-family": "'Inter', sans-serif",
    "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "min-height": "100vh",
}


def save_as_default_model(source_model_path: str) -> Dict[str, Any]:
    """선택된 모델을 기본 DDPG 모델로 저장"""
    try:
        # 기본 모델 저장 경로
        default_model_path = "./model/rl_ddpg"

        # 소스 모델 경로가 존재하는지 확인
        if not os.path.exists(source_model_path):
            return {
                "success": False,
                "message": f"소스 모델이 존재하지 않습니다: {source_model_path}",
            }

        # 필수 파일들 확인
        required_files = ["checkpoint_last.pth", "metadata_last.json"]
        missing_files = []

        for file_name in required_files:
            file_path = os.path.join(source_model_path, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)

        if missing_files:
            return {
                "success": False,
                "message": f"필수 파일이 누락되었습니다: {', '.join(missing_files)}",
            }

        # 대상 디렉토리 생성
        os.makedirs(default_model_path, exist_ok=True)

        # 기존 파일들 백업 (있는 경우)
        backup_created = False
        if os.path.exists(os.path.join(default_model_path, "checkpoint_last.pth")):
            backup_path = f"{default_model_path}_backup_{int(time.time())}"
            shutil.copytree(default_model_path, backup_path)
            backup_created = True
            logger.info(f"기존 모델 백업 생성: {backup_path}")

        # 파일들 복사
        copied_files = []
        for file_name in os.listdir(source_model_path):
            source_file = os.path.join(source_model_path, file_name)
            target_file = os.path.join(default_model_path, file_name)

            if os.path.isfile(source_file):
                shutil.copy2(source_file, target_file)
                copied_files.append(file_name)

        # 메타데이터 업데이트 (기본 모델로 저장되었음을 표시)
        metadata_file = os.path.join(default_model_path, "metadata_last.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                metadata.update(
                    {
                        "saved_as_default": True,
                        "original_source": source_model_path,
                        "default_save_time": time.time(),
                        "backup_created": backup_created,
                    }
                )

                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

            except Exception as e:
                logger.warning(f"메타데이터 업데이트 실패: {e}")

        return {
            "success": True,
            "message": f"모델이 기본 DDPG 모델로 저장되었습니다",
            "source": source_model_path,
            "target": default_model_path,
            "copied_files": copied_files,
            "backup_created": backup_created,
        }

    except Exception as e:
        error_msg = f"기본 모델 저장 중 오류 발생: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "message": error_msg}


def delete_model_folder(model_path: str) -> Dict[str, Any]:
    """모델 폴더를 안전하게 삭제"""
    try:
        # 모델 경로가 존재하는지 확인
        if not os.path.exists(model_path):
            return {
                "success": False,
                "message": f"삭제할 모델이 존재하지 않습니다: {model_path}",
            }

        # 모델 경로가 올바른 형식인지 확인 (보안 체크)
        if not os.path.basename(model_path).startswith("rl_ddpg"):
            return {
                "success": False,
                "message": f"허용되지 않은 모델 폴더입니다: {model_path}",
            }

        # 기본 모델 보호 (삭제 금지)
        protected_models = [
            "./model/rl_ddpg",
            "./model/rl_ddpg_latest",
            "model/rl_ddpg",
            "model/rl_ddpg_latest",
        ]

        # 절대 경로로 변환하여 비교
        absolute_model_path = os.path.abspath(model_path)
        for protected_path in protected_models:
            if os.path.abspath(protected_path) == absolute_model_path:
                return {
                    "success": False,
                    "message": f"기본 모델은 삭제할 수 없습니다: {os.path.basename(model_path)}",
                }

        # 메타데이터 정보 수집 (삭제 전 로깅용)
        metadata_info = get_model_metadata(model_path)
        model_name = os.path.basename(model_path)

        # 모델 폴더 내용 확인 및 로깅
        folder_contents = []
        try:
            for item in os.listdir(model_path):
                item_path = os.path.join(model_path, item)
                if os.path.isfile(item_path):
                    file_size = os.path.getsize(item_path)
                    folder_contents.append(f"{item} ({file_size:,} bytes)")
                else:
                    folder_contents.append(f"{item}/ (폴더)")
        except Exception as e:
            logger.warning(f"폴더 내용 확인 실패: {e}")

        # 삭제 전 상세 로깅
        logger.info(f"🗑️ 모델 폴더 삭제 시작: {model_path}")
        if metadata_info:
            logger.info(
                f"   📊 메타데이터: 에피소드 {metadata_info['episode']}, 날짜 {metadata_info['date']}"
            )
        if folder_contents:
            logger.info(
                f"   📁 폴더 내용: {', '.join(folder_contents[:5])}{'...' if len(folder_contents) > 5 else ''}"
            )

        # 실제 폴더 삭제 실행
        shutil.rmtree(model_path)

        success_msg = f"모델 폴더가 성공적으로 삭제되었습니다: {model_name}"
        if metadata_info:
            success_msg += f" (에피소드 {metadata_info['episode']})"

        logger.info(f"✅ 모델 폴더 삭제 완료: {model_path}")
        return {"success": True, "message": success_msg}

    except PermissionError:
        error_msg = f"폴더 삭제 권한이 없습니다: {model_path}"
        logger.error(error_msg)
        return {"success": False, "message": error_msg}
    except Exception as e:
        error_msg = f"모델 폴더 삭제 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "message": error_msg}


def get_model_deletion_info(model_path: str) -> Dict[str, Any]:
    """모델 삭제 전 상세 정보 제공"""
    try:
        if not os.path.exists(model_path):
            return {"exists": False, "message": "모델이 존재하지 않습니다"}

        # 기본 정보
        model_name = os.path.basename(model_path)
        metadata_info = get_model_metadata(model_path)

        # 폴더 크기 계산
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                except OSError:
                    pass

        # 보호된 모델인지 확인
        protected_models = [
            "./model/rl_ddpg",
            "./model/rl_ddpg_latest",
            "model/rl_ddpg",
            "model/rl_ddpg_latest",
        ]
        absolute_model_path = os.path.abspath(model_path)
        is_protected = any(
            os.path.abspath(p) == absolute_model_path for p in protected_models
        )

        return {
            "exists": True,
            "model_name": model_name,
            "model_path": model_path,
            "is_protected": is_protected,
            "metadata": metadata_info,
            "total_size": total_size,
            "file_count": file_count,
            "size_mb": round(total_size / (1024 * 1024), 2) if total_size > 0 else 0,
        }

    except Exception as e:
        logger.error(f"모델 정보 수집 중 오류: {e}")
        return {"exists": False, "message": f"모델 정보 수집 실패: {str(e)}"}
