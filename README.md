# energy_community
studying core-selecting mechanism design for local energy community

## 가상환경 설정

이 프로젝트는 상위 디렉토리의 `.venv` 폴더에 있는 Python 가상환경을 사용합니다.

### 가상환경 활성화

#### Linux/Mac:
```bash
source activate_venv.sh
```

#### Windows:
```cmd
activate_venv.bat
```

#### 수동 활성화:
```bash
source ../.venv/bin/activate  # Linux/Mac
# 또는
..\.venv\Scripts\activate.bat  # Windows
```

### VS Code 설정

VS Code에서 자동으로 가상환경을 인식하도록 설정되어 있습니다. 
`.vscode/settings.json` 파일에서 Python 인터프리터 경로가 설정되어 있습니다.

### 의존성 관리

프로젝트 의존성은 `requirements.txt` 파일에서 관리됩니다.

새로운 패키지 설치:
```bash
pip install package_name
pip freeze > requirements.txt
```

의존성 설치:
```bash
pip install -r requirements.txt
```
