# 配布データと応募用ファイル作成方法の説明

本コンペティションで配布されるデータと応募用ファイルの作成方法や投稿する際の注意点について説明する.

1. [配布データ](#配布データ)
1. [応募用ファイルの作成方法](#応募用ファイルの作成方法)
1. [投稿時の注意点](#投稿時の注意点)

## 配布データ

配布されるデータは以下の通り.

- [Readme](#readme)
- [動作確認用のプログラム](#動作確認用のプログラム)
- [応募用サンプルファイル](#応募用サンプルファイル)
- [レポートテンプレート](#レポートテンプレート)

### Readme

配布用データの説明と応募用ファイルの作成方法を説明したドキュメント.

- `readme_ja.md`: 日本語版(このファイル)
- `readme_en.md`: 英語版

マークダウン形式で, プレビューモードで見ることを推奨する.

### 動作確認用のプログラム

動作確認用プログラム一式. `run_test.zip`であり, 解凍すると以下のようなディレクトリ構造のデータが生成される.

```bash
run_test
├── finetuning                    転移学習を行うプログラムを格納したディレクトリ
│   ├── loadDB.py
│   ├── main.py
│   ├── mobilenet.py
│   ├── model_select.py
│   ├── resnet.py
│   └── train_val.py
├── pretraining                   事前学習を行うプログラムを格納したディレクトリ
│   ├── loaddata.py
│   ├── main.py
│   ├── mobilenet.py
│   ├── resnet.py
│   └── train_val.py
├── src                           検証を実行するためのプログラムを格納したディレクトリ
│   └── validator.py
├── submit                        VisualAtomにより画像を生成するためのプログラムを格納したディレクトリ
│   ├── params
│   │   └── settings.py
│   ├── src
│   │   └── generator.py
│   └── requirements.txt
├── config.yaml                   aiaccelで実行する際の設定ファイル
├── docker-compose.yml            実行環境を構築するためのcomposeファイル
├── Dockerfile                    docker環境の元となるDockerfile
├── input.json                    事前学習や転移学習時の学習パラメータ等を記載したファイル
├── LICENSE                       ライセンスファイル
├── LICENSES_bundled.txt          ライセンスファイル
├── make_submit.sh                提出用ファイルを作成するためのスクリプト
├── pretrain_finetune.py          事前学習と転移学習を実行するためのプログラム
├── run_evaluate.sh               事前学習用画像データの生成と事前学習と転移学習を実行するスクリプト
├── run.py                        事前学習用画像データを生成するプログラム
├── set_data.sh                   転移学習に使うデータセットを取得するスクリプト
└── wrapper.py                    aiaccelで実行されるプログラム
```

使い方の詳細は[応募用ファイルの作成方法](#応募用ファイルの作成方法)を参照されたい.

### 応募用サンプルファイル

応募用サンプルファイルは`sample_submit.zip`として与えられる. 解凍すると以下のようなディレクトリ構造のデータが生成される.

```bash
sample_submit
├── params             パラメータなどを置くディレクトリ
│   └── params.json
└── src                Pythonのプログラムを置くディレクトリ
    └── generator.py
```

実際に作成する際に参照されたい.

### レポートテンプレート

提出するレポートのテンプレートは`report_ja.pptx`(日本語版), `report_en.pptx`(英語版)として与えられる. 実際に作成する際に参考にされたい.

## 応募用ファイルの作成方法

パラメータファイルなどを含めた, 事前学習用画像データの生成を実行するためのソースコード一式をzipファイルでまとめたものとする.

### ディレクトリ構造

以下のようなディレクトリ構造となっていることを想定している.

```bash
.
├── params             必須: パラメータなどを置くディレクトリ
│   └── ...
├── src                必須: Pythonのプログラムを置くディレクトリ
│   ├── generator.py   必須: 最初のプログラムが呼び出すファイル
│   └── ...            その他のファイル (ディレクトリ作成可能)
└── requirements.txt   任意: 追加で必要なライブラリ一覧
```

- パラメータなどの格納場所は"params"ディレクトリを想定している.
  - 使用しない場合でも空のディレクトリを作成する必要がある.
  - 名前は必ず"params"とすること.
- Pythonのプログラムの格納場所は"src"ディレクトリを想定している.
  - パラメータ等を読み込んでデータを生成するためのメインのソースコードは"generator.py"を想定している.
    - ファイル名は必ず"generator.py"とすること.
  - その他実行するために必要なファイルがあれば作成可能である.
  - ディレクトリ名は必ず"src"とすること.
- 実行するために追加で必要なライブラリがあれば, その一覧を"requirements.txt"に記載することで, 評価システム上でも実行可能となる.
  - インストール可能で実行可能かどうか予めローカル環境で試しておくこと.
  - 評価システムの実行環境については, [*こちら*](https://hub.docker.com/layers/signate/runtime-gpu/dgm_env/images/sha256-4c7f4dc934d2067d92396cca205163c05bd9d27fb7701a0f86f67044d34a1b1c?context=explore)を参照すること.

### 環境構築

評価システムと同じ環境を用意する. Dockerイメージが[こちら](https://hub.docker.com/layers/signate/runtime-gpu/dgm_env/images/sha256-4c7f4dc934d2067d92396cca205163c05bd9d27fb7701a0f86f67044d34a1b1c?context=explore)で公開されているので, pullしてコンテナを作成して環境構築を行うことを推奨する. Dockerから環境構築する場合, Docker Engineなど, Dockerを使用するために必要なものがない場合はまずはそれらを導入しておく. [Docker Desktop](https://docs.docker.com/get-docker/)を導入すると必要なものがすべてそろうので, 自身の環境に合わせてインストーラをダウンロードして導入しておくことが望ましい. 現状, Linux, Mac, Windowsに対応している. そして, `/path/to/run_test`に同封してある`docker-compose.yml`で定義されたコンテナを, 以下のコマンドを実行することで立ち上げる.

```bash
$ cd /path/to/run_test
$ docker compose up -d
...
```

コンテナ内でインストールされている主なPythonのライブラリは以下の通り.

```bash
certifi==2023.7.22
charset-normalizer==3.2.0
cmake==3.27.0
filelock==3.12.2
h5py==3.9.0
idna==3.4
Jinja2==3.1.2
lit==16.0.6
MarkupSafe==2.1.3
mpmath==1.3.0
networkx==3.1
numpy==1.24.4
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-cupti-cu11==11.7.101
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
nvidia-cufft-cu11==10.9.0.58
nvidia-curand-cu11==10.2.10.91
nvidia-cusolver-cu11==11.4.0.1
nvidia-cusparse-cu11==11.7.4.91
nvidia-nccl-cu11==2.14.3
nvidia-nvtx-cu11==11.7.91
opencv-python==4.8.0.74
pandas==2.0.3
Pillow==10.0.0
python-dateutil==2.8.2
pytz==2023.3
requests==2.31.0
six==1.16.0
sympy==1.12
torch==2.0.1
torchvision==0.15.2
triton==2.0.0
typing_extensions==4.7.1
tzdata==2023.3
urllib3==2.0.4
```

`docker-compose.yml`は好きに編集するなりして, 自身が使いやすいように改造してもよい. GPUが使えてCUDAを有効化したい場合は以下のように編集することでコンテナ内で使用することができる.

```yaml
version: "3"
services:
  dev1:
    image: signate/runtime-gpu:dgm_env
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    container_name: analysis_dgm
    ports:
      - "8080:8080"
    volumes:
      - .:/workspace
    tty: true
```

無事にコンテナが走ったら, 必要なデータなどをコンテナへコピーする.

```bash
$ docker cp /path/to/some/file/or/dir {コンテナ名}: {コンテナ側パス}
... 
```

そして, 以下のコマンドでコンテナの中に入り, 分析や開発を行う.

```bash
$ docker exec -it {コンテナ名} bash
...
```

`コンテナ名`には`docker-compose.yml`の`services`->`dev1`->`container_name`に記載の値を記述する. デフォルトでは`/path/to/run_test`をコンテナ側の`/workspace`へバインドマウントした状態(`/path/to/run_test`でファイルの編集などをしたらコンテナ側にも反映される. 逆もしかり.)でスタートする. 追加でPythonライブラリをインストールしたい場合は例えば`requirements.txt`によりコンテナの中でインストール可能.

```bash
# コンテナに入った後
$ cd /workspace
$ pip install -r requirements.txt
...
```

CUDA環境を構築した場合, 実際にCUDAがコンテナ内で有効になっているかどうかは以下のコマンドで確認できる.

```bash
# コンテナに入った後
$ python -c "import torch; print(torch.cuda.is_available())"
True
```

### generator.pyの実装方法

以下のクラスとメソッドを実装すること.

#### Generator

データを生成するのためのクラス. 以下のメソッドを実装すること.

##### get_params

パラメータ情報などを取得するメソッド. 以下の条件を満たす必要がある.

- クラスメソッドであること.
- 引数`params_path`(str型)を指定すること.
  - パラメータ情報などが格納されているディレクトリのパス.
- パラメータの読み込みに成功した場合は`True`を返す.
  - パラメータ自体は任意の名前(例えば`params`)で保存しておく.

##### generate

データ生成を実行するメソッド. 以下の条件を満たす必要がある.

- クラスメソッドであること.
- 引数`out_path`(str型)を指定すること.
  - 生成したデータを格納するディレクトリのパス名.

実行後, 以下のように何らかのカテゴリごとにディレクトリを作成し, その下に画像データを配置するような構造で保存されるようにする.

```bash
out_path
├── category000
│   ├── image00.png(or jpg, etc.)
│   ├── image01.png(or jpg, etc.)
│   └── ...
├── category001
└── ...
```

カテゴリは1000種類作成し, それぞれの画像の枚数は1000枚とする. 参考として, VisualAtomによる画像生成アルゴリズムが`/path/to/run_test/submit`以下に実装されているので, 適宜参照すること(さらなる詳細は[こちら](https://github.com/masora1030/CVPR2023-FDSL-on-VisualAtom/tree/main)を参照されたい.). 渡すパラメータは`/path/to/params`以下にjsonファイルとしてまとめられている.

主なパラメータは以下の通り.

- `numof_thread`: 出力処理の並列数を設定する.
- `numof_classes`: データセットのクラス数を設定する.
- `numof_instances`: データセットの1クラス毎の画像枚数を設定する.
- `image_size`: 出力される画像サイズを設定する.

また, 応募用サンプルファイル`sample_submit.zip`の中身も参照されたい.

### 実行テスト

事前学習用の画像データ生成を行うプログラムが実装できたら, 正常に動作するか確認する.

以下, コンテナ内での作業とする.

#### 転移学習用のデータセットの取得

転移学習用のデータセットを取得する. 以下のコマンドで取得が可能である.

```bash
$ bash set_data.sh
...
```

デフォルトでは`CIFAR10`のデータセットが取得され, `/path/to/run_test/test/CIFAR10`以下に学習用(`train`)と評価用(`val`)が作成される. その他取得可能なデータセットについては[こちら](https://github.com/chatflip/ImageRecognitionDataset.git)を参照されたい. 他のデータセットを取得したい場合は`set_data.sh`の`DATASET`を好きな名前に変更して実行すること.

```bash
test/CIFAR10
├── train      学習用
└── val        評価用
```

#### データセット生成と事前学習と転移学習の実行

[転移学習用のデータセットの取得](#転移学習用のデータセットの取得)で取得したデータセットに対する精度評価を行う.

```bash
$ aiaccel-start --config config.yaml --clean
...
```

デフォルトのまま実行に成功すると事前学習用データセットが`/path/to/run_test/output/pretrain`以下に生成され, `/path/to/run_test/weights/pretrained`以下に事前学習済みモデルが, `/path/to/run_test/weights/CIFAR10`以下に転移学習後のモデルが保存される. 最終精度評価が出力され, `/path/to/run_test/results`以下に実行結果が保存される. `scores.json`にはデータセットごとの累積誤差(%)が記載される.

```json
{
  "CIFAR10": 30.0,
  ...
}
```

標準出力ではそれぞれのデータセットの累積誤差(%)の平均が出る. この値は小さいほど高精度である.

`/path/to/run_test/input.json`で事前学習と転移学習時に必要なパラメータや評価に使用するデータセットを設定できる. jsonファイルとなっており, 各キーは以下の通り.

- `pretrain`: 事前学習時の設定.
  - `no_cuda`: CUDA環境を使用するか否か. 使用する場合は`false`.
  - `seed`: 学習時のシード値.
  - `crop_size`: 画像をクロッピング処理する際の一辺の長さ. モデル学習時にこの大きさの画像データがモデルに渡される.
  - `weight_path`: モデルの重みの保存先.
  - `usenet`: 事前学習モデルのアーキテクチャ. 使用可能なモデルは`/path/to/run_test/pretraining/main.py`の`model_select`で確認できる. `mobilenet`を使用する際は`crop_size`は32の前提で, `resnet`の場合は224.
  - `lr`: 学習率.
  - `momentum`: 慣性項.
  - `weight_decay`: 正則化係数.
  - `resume`: checkpointのファイル名.
  - `start_epoch`: 学習を始めるエポック. 自然数.
  - `log_interval`: 学習時に損失や精度を表示する反復回数のタイミング. 自然数.
  - `save_interval`: モデルの重みを保存するエポックのタイミング. 自然数.
  - `batch_size`: 学習時のミニバッチサイズ. メモリ不足になる場合は適宜小さくするなどして調整する.
  - `epochs`: エポック数. `save_interval`よりも大きい値に設定しなければモデルの重みは保存されない.
  - `no_multigpu`: `false`の場合, 複数のGPUがあれば, それらで学習を行う.
  - `train_num_workers`: 学習時にミニバッチを作成する際に並列処理を行うときのワーカー数. 0の場合は並列処理は行われない.
  - `test_num_workers`: 検証時にミニバッチを作成する際に並列処理を行うときのワーカー数. `val`が`false`の時は有効化されない.
  - `val`: 検証を行うか否か. `true`の時は学習データで検証が行われる. 通常は`false`でよい.
  - `on_memory`: データをメモリにのせるか否か. `true`にすると条件によっては学習が高速化する.
- `finetune`: 転移学習時の設定.
  - `useepoch`: 転移学習に使用したいモデルを事前学習モデルを保存したエポック数で指定する. 存在することを事前に確認.
  - `no_cuda`: CUDA環境を使用するか否か. 使用する場合は`false`.
  - `img_size`: 学習と評価時の画像サイズ. まずこの大きさにリサイズされる.
  - `crop_size`: 画像をクロッピング処理する際の一辺の長さ. モデル学習時にこの大きさの画像データがモデルに渡される.
  - `lr`: 学習率.
  - `momentum`: 慣性項.
  - `weight_decay`: 正則化係数.
  - `resume`: checkpointのファイル名.
  - `start_epoch`: 学習を始めるエポック. 自然数.
  - `log_interval`: 学習時に損失や精度を表示する反復回数のタイミング. 自然数.
  - `save_interval`: モデルの重みを保存するエポックのタイミング. 自然数.
  - `train_batch_size`: 学習時のミニバッチサイズ. メモリ不足になる場合は適宜小さくするなどして調整する.
  - `test_batch_size`: 評価時のミニバッチサイズ. メモリ不足になる場合は適宜小さくするなどして調整する.
  - `epochs`: エポック数. `save_interval`よりも大きい値に設定しなければモデルの重みは保存されない.
  - `no_multigpu`: `false`の場合, 複数のGPUがあれば, それらで学習を行う.
  - `seed`: 学習時のシード値.
  - `train_num_workers`: 学習時にミニバッチを作成する際に並列処理を行うときのワーカー数. 0の場合は並列処理は行われない.
  - `test_num_workers`: 検証時にミニバッチを作成する際に並列処理を行うときのワーカー数. 0の場合は並列処理は行われない.
  - `on_memory`: データをメモリにのせるか否か. `true`にすると条件によっては学習が高速化する.
  - `min_result_epochs`: 誤差率の計算を始めるエポックのタイミング
- `datasets`: 転移学習時に使用するデータセット名のリスト
  - 転移学習用のデータセットの一覧. [ここ](#転移学習用のデータセットの取得)で取得したデータセット名をリストで記載する.

特に使用メモリや実行時間を調整したい場合は各種パラメータを変更して実行されたい. なお, 評価システム上の設定に関しては非公開となる.

投稿する前にエラーが出ずに実行が成功することを確認すること.

### 応募用ファイルの作成

上記の[ディレクトリ構造](#ディレクトリ構造)となっていることを確認して, zipファイルとして圧縮する. 以下のコマンドで応募用ファイルを作成可能.

```bash
$ bash make_submit.sh
...
```

実行後, 作業ディレクトリにおいて`submit.zip`が作成される.

## 投稿時の注意点

- 評価システム上ではインターネット接続を行えないので, 提出プログラム内にインターネット接続を行うような処理は実装しないこと.
- 投稿後, 結果が返ってくるまでに1.6~2.0日程度かかる想定である.
- 事前学習用画像データセットの生成にかけられる時間は12時間(CPU: コア数=40, メモリ=360[GB])が上限となる.

## 参考文献

1. Kataoka, Hirokatsu and Okayasu, Kazushige and Matsumoto, Asato and Yamagata, Eisuke and Yamada, Ryosuke and Inoue, Nakamasa and Nakamura, Akio and Satoh, Yutaka. Pre-training without Natural Images. In International Journal on Computer Vision (IJCV), 2022
1. Kataoka, Hirokatsu and Okayasu, Kazushige and Matsumoto, Asato and Yamagata, Eisuke and Yamada, Ryosuke and Inoue, Nakamasa and Nakamura, Akio and Satoh, Yutaka. Pre-training without Natural Images. In Asian Conference on Computer Vision (ACCV), 2020
1. Takashima, Sora and Hayamizu, Ryo and Inoue, Nakamasa and Kataoka, Hirokatsu and Yokota, Rio. Visual Atoms: Pre-training Vision Transformers with Sinusoidal Waves. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023
