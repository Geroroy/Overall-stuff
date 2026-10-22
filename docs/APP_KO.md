# FBMOD Converter 사용법

`.fbmod` 파일을 넣으면 바로 3D 모델(FBX, GLB)과 텍스처로 변환해 주는 프로그램입니다.

## 1. 준비

1. **FBMOD-Converter.exe**를 받습니다.
   GitHub 저장소의 **Actions** 탭 → **Build FBMOD Converter (Windows)** → 최근 실행 → 아래쪽
   **Artifacts**의 `FBMOD-Converter-windows`를 받아 압축을 풀면 됩니다.
   (직접 빌드하려면 Python 설치 후 저장소 폴더에서 `packaging\build_windows.bat` 실행)
2. **Blender**(4.2 이상)를 설치합니다. FBX를 만들 때 필요하며, 기본 설치 위치에 있으면
   프로그램이 자동으로 찾습니다. Blender가 없어도 GLB 파일(Blender에서 바로 열림)과 텍스처는 만들어집니다.

## 2. 변환

다음 중 편한 방법을 쓰세요. 파일을 넣는 즉시 변환이 시작됩니다.

* `.fbmod` 파일을 **FBMOD-Converter.exe 아이콘 위에 끌어다 놓기**
* 프로그램을 실행하고 창의 파란 영역에 **파일을 끌어다 놓기**
* 파란 영역을 **클릭해서 파일 선택**

끝나면 결과 요약 창이 뜹니다. **결과 폴더 열기**를 누르면 됩니다.

## 3. 결과물

모드 파일 옆에 `모드이름_converted` 폴더가 생깁니다. (저장 위치를 지정했다면 그곳)

| 폴더/파일 | 내용 |
|---|---|
| `fbx\` | **FBX 모델** (텍스처 포함) |
| `meshes\` | GLB 모델 (Blender: File → Import → glTF) |
| `blender\` | Blender 파일(.blend) |
| `Extracted_Textures\` | 텍스처 (PNG, 원본 DDS, 컬러/러프니스/메탈릭/노멀 분리본) |
| `Extraction_Report.json` | 무엇이 변환됐고 무엇이 빠졌는지 상세 리포트 |

## 4. 고급 설정 (선택)

창의 **고급 설정**을 체크하면 나옵니다.

* **출력 이름**: `body=Anakin_Body, head=Anakin_Head`처럼 쓰면, 이름에 `body`가 들어간 메시는
  `Anakin_Body.fbx`로 저장됩니다.
* **모든 LOD 내보내기**: 저품질 LOD까지 모두 내보냅니다. (기본은 최고 품질만)
* **게임 데이터 폴더**: 모드에 없는 원본 게임 데이터(머리 메시, 바꾸지 않은 텍스처 등)를
  Frosty Editor로 내보내 둔 폴더. 무엇을 내보내야 하는지는 결과 요약/리포트에 나옵니다
  ([자세한 방법](MISSING_DEPENDENCIES.md)).
* **스켈레톤 파일**: 캐릭터의 SkeletonAsset을 Frosty Editor에서 Binary(.bin)로 내보낸 파일.
  지정하면 FBX에 본(뼈대)과 스킨 웨이트가 들어갑니다.

설정은 다음 실행 때도 기억됩니다.

## 5. 자주 나오는 메시지

* **"변환할 3D 메시가 없습니다"**: 그 모드에 모델 데이터가 들어 있지 않습니다
  (예: 기존 모델의 설정만 바꾸는 모드). 프로그램 오류가 아닙니다.
* **"게임 원본 데이터가 필요한 항목"**: 모드가 일부만 바꿨기 때문에 나머지는 게임 파일에만 있습니다.
  리포트의 `missing_dependencies`에 무엇을 Frosty Editor로 내보내야 하는지 적혀 있습니다.
* **"스켈레톤이 모드에 없어서 FBX에는 본/웨이트가 빠집니다"**: 모드에는 보통 스켈레톤이 없습니다.
  위의 스켈레톤 파일을 지정하면 해결됩니다. (.blend와 .glb에는 웨이트가 보존되어 있습니다)
* **Windows에서 "PC 보호" 경고**: 서명되지 않은 프로그램이라 처음 실행 시 뜰 수 있습니다.
  "추가 정보 → 실행"을 누르세요.
