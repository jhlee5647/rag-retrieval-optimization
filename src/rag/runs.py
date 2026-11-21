"""실행 폴더."""

import shutil
from datetime import datetime
from pathlib import Path


def create_run_dir(root: Path, config_path: Path) -> Path:
    """`<root>/<config 이름>/<실행 시각>/`을 만들고 config 사본을 넣는다.

    Args:
        root: 명령별 실행 폴더의 상위 폴더. 예: `runs/retrieve`.
        config_path: 실행에 쓴 config 파일. 확장자를 뺀 이름이 폴더 이름이 된다.

    Returns:
        새로 만든 실행 폴더.

    Raises:
        FileExistsError: 같은 초에 같은 config로 실행해 폴더가 이미 있는 경우.
    """
    run_dir = root / config_path.stem / datetime.now().strftime("%Y%m%d-%H%M%S")
    # 앞선 실행의 결과를 덮어쓰지 않도록, 이미 있으면 실패한다.
    run_dir.mkdir(parents=True, exist_ok=False)
    # 주석과 적힌 순서까지 남도록, 읽은 값을 다시 쓰지 않고 파일을 그대로 복사한다.
    shutil.copyfile(config_path, run_dir / "config.yaml")
    return run_dir
