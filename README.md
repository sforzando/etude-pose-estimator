# etude-pose-estimator

MediaPipe Pose Landmarkerを使った姿勢推定プロトタイプ

## 概要

このアプリケーションは、MediaPipe Pose Landmarkerを使用して画像から姿勢を検出し、関節角度を計算します。また、基準ポーズとの類似度をProcrustes解析で比較できます。

## 機能

### バックエンド
- **FastAPI + Jinja2テンプレート**: 高速なWebフレームワーク
- **MediaPipe Pose Landmarker**: 33キーポイントで単一画像から姿勢検出
- **関節角度算出**: 肘、肩、膝の角度を自動計算
- **Procrustes解析**: 基準ポーズとの類似度を数値化
- **基準ポーズ登録**: キーポイントをJSONで保存・管理

### フロントエンド
- **htmx**: 軽量なインタラクション（画像アップロード、結果表示）
- **Tailwind CSS + daisyUI**: モダンで美しいUIコンポーネント
- **レスポンシブデザイン**: モバイルファーストの設計

## 技術スタック

- **Python 3.12**: 最新のPython機能を活用
- **uv**: 高速な依存パッケージ管理
- **Ruff**: 高速なリンター＆フォーマッター
- **Taskfile**: タスクランナー
- **direnv**: ローカル環境変数管理
- **Google Cloud Run**: コンテナベースのデプロイ

## セットアップ

### 前提条件

- Python 3.12以上
- uv（推奨）またはpip
- Task（タスクランナー）
- direnv（オプション）

### インストール

1. リポジトリをクローン:
```bash
git clone https://github.com/sforzando/etude-pose-estimator.git
cd etude-pose-estimator
```

2. uvで依存関係をインストール:
```bash
uv pip install -e ".[dev]"
```

または、Taskfileを使用:
```bash
task install
```

3. MediaPipe Pose Landmarkerモデルをダウンロード:
```bash
# ライトモデル（推奨）
curl -L -o pose_landmarker.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

# または、フルモデル（より高精度）
curl -L -o pose_landmarker.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
```

**注**: ネットワーク環境によってはダウンロードが制限される場合があります。その場合は、ブラウザで上記URLにアクセスしてファイルを手動でダウンロードし、プロジェクトルートに `pose_landmarker.task` として配置してください。

4. 環境変数を設定（オプション）:
```bash
cp .envrc.example .envrc.local
direnv allow
```

## 使い方

### 開発サーバーの起動

```bash
task dev
```

または直接:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

ブラウザで http://localhost:8000 にアクセス

### 基準ポーズの登録

1. 「基準ポーズ登録」タブを開く
2. ポーズ名を入力（例: yoga_tree_pose）
3. 基準となる姿勢の画像をアップロード
4. 「登録する」ボタンをクリック

### 姿勢分析

1. 「姿勢分析」タブを開く
2. 分析したい姿勢の画像をアップロード
3. （オプション）基準ポーズを選択
4. 「分析する」ボタンをクリック
5. 結果を確認:
   - 関節角度（肘、肩、膝）
   - 類似度スコア（基準ポーズ選択時）
   - 角度差分（基準ポーズ選択時）

## 開発

### リンティング

```bash
task lint
```

### フォーマット

```bash
task format
```

### 自動修正

```bash
task lint-fix
```

### クリーンアップ

```bash
task clean
```

## デプロイ

### Google Cloud Runへのデプロイ

1. Dockerイメージをビルド:
```bash
docker build -t etude-pose-estimator .
```

2. Google Cloud Runにデプロイ:
```bash
gcloud run deploy etude-pose-estimator \
  --source . \
  --platform managed \
  --region asia-northeast1 \
  --allow-unauthenticated
```

## プロジェクト構成

```
.
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPIアプリケーション
│   ├── pose_estimator.py    # MediaPipe Pose Landmarker統合
│   ├── pose_analyzer.py     # 関節角度計算とProcrustes解析
│   └── templates/
│       ├── index.html       # メインページ
│       ├── result.html      # 分析結果表示
│       └── register_result.html  # 登録結果表示
├── reference_poses/         # 基準ポーズのJSON保存先
├── uploads/                 # 一時的なアップロードファイル
├── pyproject.toml          # プロジェクト設定＆依存関係
├── Taskfile.yml            # タスク定義
├── Dockerfile              # Cloud Run用Dockerfile
├── .gitignore              # Git無視ファイル
└── README.md               # このファイル
```

## アーキテクチャ

### MediaPipe Pose Landmarker

33個のキーポイントで全身の姿勢を検出:
- 顔（目、鼻、耳、口）
- 上半身（肩、肘、手首、手）
- 下半身（腰、膝、足首、足）

### 関節角度計算

3点から角度を計算:
- 肘: 肩-肘-手首
- 肩: 肘-肩-腰
- 膝: 腰-膝-足首

### Procrustes解析

2つの姿勢の類似度を計算:
1. 座標の正規化
2. 最適な回転・スケーリングを適用
3. 残差から類似度スコア（0-100%）を算出

## ライセンス

MIT License

## 作者

sforzando