# GEM-4-VLA

**SigLIP + Gemma-4-E2B + ドメイン別プロジェクタ + L1 action head** をベースにした
ウェアラブル Vision-Language-Action アシスタント。LIBERO ベンチマークと、
MimicRec 経由の実機デプロイをターゲットにしています。

プロジェクトの背景、ウェアラブルシステム全体の構成、MimicRec / MimicAnno
といった周辺ツール、研究上の位置づけについては Kaggle 記事を参照してください。
**この README は本リポジトリで配布している VLA の学習・評価・推論パイプラインを
再現するための手順書です。**

English version: [README.md](README.md)

## 結果

すべての数値は **10 episode / task × 10 task = 100 ep / suite** (`eval.num_episodes_per_task: 10`、
headless MuJoCo) で取得。LIBERO 4 suite はすべて、共通の事前学習ベース
(OXE 9 dataset + LIBERO 4-suite mix, `step_100000`) からファインチューニングしています。

### LIBERO 4-suite (FT step_50000)

| suite     | success rate | HF checkpoint |
|-----------|-------------:|---|
| spatial   | **72 %** | [`takaki99/GEM-4-FT-libero-spatial`](https://huggingface.co/takaki99/GEM-4-FT-libero-spatial) |
| object    | **92 %** | [`takaki99/GEM-4-FT-libero-object`](https://huggingface.co/takaki99/GEM-4-FT-libero-object)   |
| goal      | **89 %** | [`takaki99/GEM-4-FT-libero-goal`](https://huggingface.co/takaki99/GEM-4-FT-libero-goal)       |
| 10 (long) | **43 %** | [`takaki99/GEM-4-FT-libero-10`](https://huggingface.co/takaki99/GEM-4-FT-libero-10)           |
| **平均**  | **74 %** | — |

事前学習ベース: [`takaki99/GEM-4-Pretrained-OXE`](https://huggingface.co/takaki99/GEM-4-Pretrained-OXE)
(`step_100000`)。FT レシピは spatial / object / goal が `bs=8 × 2 GPU × accum=2 = eff bs 32`、
libero_10 は `bs=8 × 4 GPU × accum=4 = eff bs 128`。

### ReBotArm FT

`GEM-4-Pretrained-OXE` の上に、ドメイン行を1つ追加してシングルタスク FT
(`num_domains: 13 → 14`、`resume_da_row_init: random`。詳細は
[CLAUDE.md の DA-row ルール](CLAUDE.md#da-row-init-for-ft-do-not-copy))：

| タスク               | dataset                                                                                              | best ckpt   | HF checkpoint                                                                            |
|----------------------|------------------------------------------------------------------------------------------------------|-------------|------------------------------------------------------------------------------------------|
| pick up the bottle   | [`takaki99/GEM4_pick_up_bottle`](https://huggingface.co/datasets/takaki99/GEM4_pick_up_bottle)       | step_30000  | [`takaki99/GEM-4-FT-bottle`](https://huggingface.co/takaki99/GEM-4-FT-bottle)            |
| open the jar         | HF 上の dataset                                                                                       | step_15000  | [`takaki99/GEM-4-FT-jar`](https://huggingface.co/takaki99/GEM-4-FT-jar)                  |

## セットアップ

`envs/` 配下に uv の環境を2つ用意しています。ホストの CPU アーキで使い分けます。

| ホスト                                | スクリプト                       | env dir         | wheels                                  |
|---------------------------------------|----------------------------------|-----------------|-----------------------------------------|
| x86_64 Linux (training / research)    | `bash scripts/setup_x86.sh`      | `envs/x86`      | PyTorch cu128 (driver ≥ 12.6)           |
| Jetson Orin (JetPack 6 / CUDA 12.6)   | `bash scripts/setup_jetson.sh`   | `envs/jetson`   | jetson-ai-lab JP6/cu126 (sm_87, cp310)  |

各 setup スクリプトは、uv のインストール (未導入なら) → `VLA-Adapter` と `X-VLA`
サブモジュールの init → `uv sync --project envs/<env>` → torch + Gemma-4 の
smoke check までを実施します。

セットアップ後は、すべてのコマンドで `--project` を指定して環境を切り替えます。

```bash
uv run --project envs/x86    python scripts/train.py configs/train/<config>.yaml
uv run --project envs/jetson python scripts/serve.py ...
```

スクリプトが **処理しない** ホスト固有の前提:

- 兄弟ディレクトリの `vla-gemma-4/` チェックアウト (RLDS データとベースライン ckpt 用、
  OXE 事前学習を再現する場合のみ必要)
- LIBERO シミュレータ + assets (`MUJOCO_GL=osmesa` で headless レンダリング)
- Gemma-4 / SigLIP 用の Hugging Face トークン
  (`uv run --project envs/<env> huggingface-cli login`)

### なぜ環境を2つに分けているか

- `tensorflow-addons==0.23.0` (`dlimp`/OXE-RLDS の transitive 依存) には
  Linux aarch64 wheel が存在しないため、Jetson 側では RLDS パイプラインの
  依存を落としています。
- upstream の PyTorch cu126/cu128/cu130 wheel は sm_90+ ビルドで、Orin (sm_87)
  では `.to('cuda')` で `no kernel image` クラッシュします。Jetson 側は
  jetson-ai-lab の JP6 / cu126 index から torch / torchvision を取得します。
- 両環境とも Python 3.10 固定です (jetson-ai-lab が cp310 のみ公開しているため)。

## 結果の再現手順

このリポジトリは **`GEM-4-Pretrained-OXE` を Hugging Face から起点に使う前提** で
書かれています。スクラッチからの事前学習 (OXE 9 + LIBERO 4 mix、~100k step) も
コード上はサポートしていますが、本 README には載せていません。必要であれば
メンテナに相談してください。

### 1. 事前学習ベースをダウンロード

```bash
mkdir -p outputs/oxe_pretrain_v47_arch_v3_libero_dl50_bs8/checkpoints
uv run --project envs/x86 huggingface-cli download \
  takaki99/GEM-4-Pretrained-OXE \
  --local-dir outputs/oxe_pretrain_v47_arch_v3_libero_dl50_bs8/checkpoints/step_100000
```

(出力ディレクトリ名は、既存 FT config の `resume_ckpt:` がそのまま参照している
パスに合わせています。ゼロ編集で再現するため、このまま使ってください。)

### 2. LIBERO suite FT

各 suite に対応する FT config が用意されています。

```bash
# libero_spatial / libero_object / libero_goal / libero_10 から選択
CONFIG=configs/train/libero_spatial_v47_step100k_ft_dl41_2gpu.yaml

CUDA_VISIBLE_DEVICES=0,1 \
  uv run --project envs/x86 accelerate launch \
    --config_file configs/accelerate/dl50_4gpu.yaml \
    --main_process_port 29501 \
    scripts/train.py $CONFIG
```

libero_10 は `configs/train/libero_10_v47_step95k_ft_4gpu_accum4.yaml` を
使います (4 GPU で effective batch 128)。

checkpoint は `outputs/<wandb.name>/checkpoints/step_<N>/` に保存されます。

### 3. ReBotArm hand-teach FT (HF dataset → FT パイプライン)

end-to-end の yaml-driven ランチャー
[`scripts/ft_lerobot_from_hf.py`](scripts/ft_lerobot_from_hf.py): HF ダウンロード
→ v3→v2.1 変換 → norm stats 算出 → 224×224 uint8 フレーム展開 → (オプション)
ローカル SSD への rsync → accelerate launch まで一気通貫。各ステップは
冪等で、出力が既に存在すればスキップします。

```bash
# 1. example yaml をコピーして、prep.hf.repo_id, dataset_key, domain_id などを編集
cp configs/train/_example_ft_from_hf.yaml configs/train/<your_ft>.yaml
$EDITOR configs/train/<your_ft>.yaml

# 2. 実行プランを dry-run で確認 (何も実行しない)
uv run --project envs/x86 python scripts/ft_lerobot_from_hf.py \
  configs/train/<your_ft>.yaml --dry_run

# 3. 本実行
uv run --project envs/x86 python scripts/ft_lerobot_from_hf.py \
  configs/train/<your_ft>.yaml
```

通常の train config に加えて、以下の2ブロックを yaml に書きます。

```yaml
prep:
  hf:
    repo_id: takaki99/GEM4_pick_up_bottle
  norm_stats:
    dataset_key: <key>
  frames:
    pre_extract: true
    workers: 16
    local_copy:                     # optional, NFS read contention 対策
      enabled: true
      host: dl42
      path: /var/tmp/<key>_frames_uint8

launch:
  host: dl42                        # null でローカル実行
  cuda_visible_devices: "0,1,2,3"
  num_processes: 4
  main_process_port: 29516
  accelerate_config: configs/accelerate/dl50_4gpu.yaml
```

ハイパラ (lr、freeze、batch など) はすべて yaml 側に書きます。CLI フラグは
動作モード制御のみ (`--dry_run`、`--no_launch`、`--force_convert / _stats /
_extract / _local`)。

## 評価

```bash
uv run --project envs/x86 python scripts/eval.py configs/eval/<your_eval>.yaml
```

エピソード数は eval yaml の `eval.num_episodes_per_task` で指定します。
上の結果テーブルで使った数値の前提:

- **5 ep / task** — FT 中に多数の step checkpoint をざっと sweep して、
  有望な ckpt 帯を見つけるための高速確認。
- **10 ep / task** — 確定値の確認。公開する数値、HF カードに載せる数値は
  必ずこのモードで取ります。5ep の値は variance の大きいタスク
  (spatial task_5、libero_10 task_8 など) で ±10 pt 程度ぶれます。

出力先:

- `outputs/<run>/eval_step<K>[<suffix>].log` — `[eval] metrics={...}` 形式の
  メトリクスがインラインで出力されます。
- `outputs/<run>/eval_videos_step<K>[<suffix>]/` — エピソード単位の MP4。

## 推論サーバ

MimicRec の `POST /predict` 契約に従う FastAPI サーバで、ckpt を読み込んで
ホストします。サーバはモデルネイティブで、q99 denormalize 済みの action chunk
を返します。フレーム変換、グリッパ規約のマッピング、生 proprio の整形は
クライアント側の責務です。

predictor は2種類:

- **`hold_position`** — 固定 action chunk を返すだけのスタブ。GPU も ckpt も
  不要なので、wire format の smoke test に使います。
- **`xvla_adapter`** (デフォルト) — 実 ckpt を読み込んで forward を回します。

```bash
# HoldPosition smoke (GPU 不要)
uv run --project envs/x86 python scripts/serve.py --predictor hold_position --port 8001
curl http://127.0.0.1:8001/healthz

# 実 ckpt を HF から直接ロード
CUDA_VISIBLE_DEVICES=0 \
  uv run --project envs/x86 python scripts/serve.py \
    --checkpoint takaki99/GEM-4-FT-bottle \
    --port 8001
```

`--checkpoint` はローカルディレクトリ、または HF repo id (`org/repo` ないし
`org/repo/subfolder`) を受け付けます。HF からの解決は
`~/.cache/huggingface/hub/` にキャッシュされ、2回目以降は無料です。
ckpt に同梱された `post_process.py` を HF 解決時に有効化したい場合は
`--trust-checkpoint-code` を明示的に渡してください。

1 リクエストあたりのレイテンシは、RTX 6000 Ada 1枚 + bf16 + `torch_compile: off`
で ~220 ms (予算 266 ms、超過時は warning ログ)。

deploy yaml の書き方、デプロイ時のグリッパ正規化、`POST /predict` の完全な
スキーマ、ランタイムの既知制約については
[`src/vla_project/deployment/README.md`](src/vla_project/deployment/README.md)
を参照してください。

## リポジトリ構成

canonical なレイアウトと coding rules は [`CLAUDE.md`](CLAUDE.md) を参照。
TL;DR:

```
src/vla_project/
  data/          # dataset → 内部 batch schema (RLDS, LeRobot, LIBERO, lerobot_preextracted)
  models/        # vision, language, projectors, action heads, vla_policy
  policies/      # runtime obs → action ラッパ
  training/      # trainer, optim, schedulers, checkpoint, distributed
  evaluation/    # libero_eval, rollout, metrics
  robots/        # base / sim / lerobot I/O
  deployment/    # serve, predictors, gripper_normalizer
configs/
  train/         # アーキ改訂 × FT recipe ごとに1 yaml
  eval/          # (ckpt × suite × step) ごとに1 yaml
  accelerate/    # ホストごとの yaml プリセット
scripts/
  train.py
  eval.py
  serve.py
  ft_lerobot_from_hf.py   # HF dataset → FT 1コマンドランチャー (yaml-driven)
tools/
  push_ckpt_to_hf.py      # ckpt dir → HF repo (optimizer 同梱 / dry-run 対応)
  extract_lerobot_frames.py
  compute_norm_stats_so101.py
  convert_rebot_bottle_v3_to_v21.py
docs/architectures/        # mermaid diagram (現行アーキ + ablations)
```

アーキ改訂はすべて `configs/train/` 配下の config ファイルとして管理しており、
コード側に散在させません。現行モデルのレイアウト (LLM 入力ストリーム、
action head の cross-attn ストリーム、projector 構成) は `docs/architectures/`
を参照してください。

## 開発

```bash
PYTHONPATH="" uv run --project envs/x86 pytest -v        # テスト
uv run --project envs/x86 ruff check src/ tests/         # lint
```

コーディングルールと貢献フローは [`DEVELOPMENT.md`](DEVELOPMENT.md) と
[`CLAUDE.md`](CLAUDE.md) を参照してください。
