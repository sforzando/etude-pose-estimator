# etude-pose-estimator

ヒーローショーの決めポーズ品質管理システム（étude版）

## 概要

スマホで撮影したポーズ動画/画像をサーバーで解析し、基準ポーズとの類似度を判定するWebアプリケーション。

## 要件

- 動画・静止画入力対応
- 体格差・カメラ角度を考慮した比較（3D姿勢推定）
- 処理時間：10分以内許容

## 技術選定

### 2D姿勢推定: YOLO11x Pose

- 17キーポイント（COCO形式）
- ウルトラマン着ぐるみでも82%信頼度で検出成功（実証済み）
- MediaPipeは着ぐるみを認識できず不採用

### 2D→3D Lifting: MotionBERT

- 2D座標から3D座標を推定
- カメラ角度の違いを吸収するために必須
- 17キーポイント（H36M形式、COCO互換）

### ポーズ比較: scipy.spatial.procrustes

- 3D座標でProcrustes解析
- 位置・スケール・回転の違いを自動除去

### アドバイス生成: Gemini 3 Flash

- `google-genai` パッケージ
- モデル: `gemini-3-flash-preview`
- 関節角度の差分から改善アドバイスを日英で生成

### フロントエンド

- htmx + Tailwind CSS + daisyUI
- ビルドステップなし、モバイルファースト

### インフラ

- Google Cloud Run with GPU（NVIDIA L4）
- Docker（nvidia/cuda:12.2.2-runtime-ubuntu22.04 + Python 3.12）

## 処理パイプライン

```plain
入力画像 → YOLO11x(2D) → MotionBERT(3D) → Procrustes比較 → Gemini 3(アドバイス) → 結果表示
```

## ディレクトリ構成

```plain
etude-pose-estimator/
├── CLAUDE.md
├── Taskfile.yml
├── pyproject.toml
├── Dockerfile
├── src/etude_pose_estimator/
│   ├── main.py
│   ├── config.py
│   ├── api/routes.py
│   ├── core/
│   │   ├── pose_2d.py      # YOLO11x
│   │   ├── pose_3d.py      # MotionBERT
│   │   ├── compare.py      # Procrustes
│   │   ├── angle.py        # 関節角度算出
│   │   ├── advice.py       # Gemini 3
│   │   └── reference.py    # 基準ポーズ管理
│   ├── templates/
│   ├── static/
│   └── i18n/
├── data/references/        # 基準ポーズJSON
├── models/                 # 事前ダウンロードモデル
└── tests/
```

## API設計

| エンドポイント | 機能 |
| --------------- | ------ |
| POST /api/detect | 画像から3D姿勢推定 |
| POST /api/compare | 基準ポーズと比較＋アドバイス生成 |
| POST /api/reference | 基準ポーズ登録 |
| GET /api/references | 基準ポーズ一覧 |

## 環境変数

- `GEMINI_API_KEY`: Gemini API キー

## 依存パッケージ

- fastapi, jinja2
- ultralytics (YOLO11)
- google-genai (Gemini 3)
- scipy, opencv-python, numpy, torch
- pydantic-settings, python-multipart

## ライセンス注意

- YOLO11: AGPL-3.0（商用はEnterprise License必要）
- MotionBERT: MIT License

## MotionBERTセットアップ

手動でのセットアップが必要:

1. <https://github.com/Walter0807/MotionBERT> からモデルコードを取得
2. チェックポイントをダウンロード
