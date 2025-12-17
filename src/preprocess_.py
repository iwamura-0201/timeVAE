from pathlib import Path
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import csv
from typing import List, Union
#from logparser.Drain import LogParser as Original_Drain
# from src.logparser.Spell import LogParser as Spell

INTERIM_DIR = Path('../data/interim')
PROCESSED_DIR = Path('../data/processed')
RAW_DIR = Path('../data/raw')


# --------------------------------------------- ここから モデル前データ作成 関連--------------------------------------------------#

def sliding_window(
    raw_data: pd.DataFrame,
    para: dict,
    mode: str,  # "time" or "fixed"
) -> pd.DataFrame:
    """
    Split logs into sliding windows.

    Parameters
    ----------
    raw_data : pd.DataFrame
        columns = [timestamp, label, eventid, time duration]
        ※列順は元コードと同じ想定

    para : dict
        mode="time" の場合:
            {
                "window_size": float,  # 1ウィンドウの時間幅 [秒]
                "step_size"  : float,  # ウィンドウを進める時間 [秒]
            }
        mode="fixed" の場合:
            {
                "window_size": int,    # 1ウィンドウに含めるイベント数（固定長）
                "step_size"  : int,    # 次のウィンドウへ進むときに何イベントずらすか
            }

    mode : str
        "time"  : 時間ベースのスライディングウィンドウ
        "fixed" : イベント数ベース（固定長）のスライディングウィンドウ

    Returns
    -------
    pd.DataFrame
        columns = raw_data.columns
        各セルには配列（timestamp列は時刻配列、labelはウィンドウラベル、など）が入る。
    """
    if raw_data.shape[0] == 0:
        raise ValueError("raw_data is empty")

    # 列の取り方は元コードと同じ
    time_data   = raw_data.iloc[:, 0]
    label_data  = raw_data.iloc[:, 1]
    logkey_data = raw_data.iloc[:, 2]
    deltaT_data = raw_data.iloc[:, 3]

    new_data = []

    # -----------------------------
    # 1) 時間ベースのウィンドウ
    # -----------------------------
    if mode == "time":
        window_size = float(para["window_size"])
        step_size   = float(para["step_size"])

        if window_size <= 0 or step_size <= 0:
            raise ValueError("window_size and step_size must be positive.")

        log_size = len(time_data)
        start_end_index_pair = []

        # time_data は昇順（時系列順）を前提
        start_index = 0
        num_session = 0

        while start_index < log_size:
            start_time = time_data.iloc[start_index]
            end_time   = start_time + window_size

            # end_index を時間で伸ばす
            end_index = start_index
            while end_index < log_size and time_data.iloc[end_index] < end_time:
                end_index += 1

            if start_index != end_index:
                start_end_index_pair.append((start_index, end_index))
                num_session += 1
                if num_session % 1000 == 0:
                    print(f"process {num_session} time window", end="\r")

            # 次のウィンドウの開始時刻
            next_start_time = start_time + step_size

            # next_start_time 以降の最初のインデックスを探す
            new_start_index = start_index
            while new_start_index < log_size and time_data.iloc[new_start_index] < next_start_time:
                new_start_index += 1

            # 念のため無限ループ回避（すべて同じ時刻などの変なケース）
            if new_start_index <= start_index:
                new_start_index = start_index + 1

            start_index = new_start_index

        # ウィンドウごとのデータを作成
        for start_index, end_index in start_end_index_pair:
            ts_seq    = time_data[start_index:end_index].values
            label_seq = label_data[start_index:end_index].values
            key_seq   = logkey_data[start_index:end_index].values
            dt_seq    = deltaT_data[start_index:end_index].values.copy()
            dt_seq[0] = 0  # 先頭は0

            window_label = label_seq.max()

            new_data.append([
                ts_seq,
                window_label,
                key_seq,
                dt_seq,
            ])

        print(
            "there are %d instances (sliding windows) in this dataset\n"
            % len(new_data)
        )
        return pd.DataFrame(new_data, columns=raw_data.columns)

    # -----------------------------
    # 2) イベント数ベース（固定長）
    # -----------------------------
    elif mode == "fixed":
        window_size = int(para["window_size"])
        step_size   = int(para.get("step_size", window_size))  # 指定なければ非オーバーラップ

        if window_size <= 0 or step_size <= 0:
            raise ValueError("window_size and step_size must be positive integers")

        log_size = raw_data.shape[0]
        num_session = 0

        for start_index in range(0, log_size - window_size + 1, step_size):
            end_index = start_index + window_size

            ts_seq    = time_data[start_index:end_index].values
            label_seq = label_data[start_index:end_index].values
            key_seq   = logkey_data[start_index:end_index].values
            dt_seq    = deltaT_data[start_index:end_index].values.copy()
            dt_seq[0] = 0

            window_label = label_seq.max()

            new_data.append([
                ts_seq,
                window_label,
                key_seq,
                dt_seq,
            ])

            num_session += 1
            if num_session % 1000 == 0:
                print(f"process {num_session} count window", end="\r")

        print(
            "there are %d instances (sliding windows) in this dataset\n"
            % num_session
        )
        return pd.DataFrame(new_data, columns=raw_data.columns)

    else:
        raise ValueError("mode must be either 'time' or 'count'.")


def npz_file_generator(
    filename: str, 
    df: pd.DataFrame, 
    features: list
) -> None:
    """
    データフレームを timeVAE 用の .npz ファイルに変換して保存する関数。
    形式: (N, T, D)
    N: サンプル数
    T: シーケンス長
    D: 特徴量数 (len(features))
    """
    data_list = []
    
    for _, row in df.iterrows():
        # 各行の指定featureを取り出し、転置して (T, D) の形にする
        # row[features] は各カラムがリスト(長さT)を持っている想定
        # zip(*row[features]) で (val_feat1, val_feat2, ...) のタプルのリスト(長さT)になる
        sequence = list(zip(*row[features]))
        data_list.append(sequence)
        
    # numpy array に変換: (N, T, D)
    # 注意: 全てのシーケンス長 T が同じである必要がある
    data_np = np.array(data_list)
    
    # 保存 (.npz 拡張子を補完するかは呼び出し元次第だが、ここではそのまま保存)
    # timeVAE は key="data" を求めている
    np.savez(filename, data=data_np)

def calculate_seq_length_stats(df: pd.DataFrame, seq_column: str = "EventId") -> dict:
    """
    DataFrameの指定カラムからシーケンス長統計を計算する。
    
    Parameters
    ----------
    df : pd.DataFrame
        シーケンスデータを含むDataFrame
    seq_column : str
        シーケンス（配列）が格納されているカラム名
    
    Returns
    -------
    dict
        統計情報の辞書 {"count", "avg_len", "min_len", "max_len", "std_len"}
    """
    if len(df) == 0:
        return {"count": 0, "avg_len": 0.0, "min_len": 0, "max_len": 0, "std_len": 0.0}
    
    lengths = df[seq_column].apply(lambda x: len(x) if hasattr(x, '__len__') else 0)
    return {
        "count": len(df),
        "avg_len": float(lengths.mean()),
        "min_len": int(lengths.min()),
        "max_len": int(lengths.max()),
        "std_len": float(lengths.std()),
    }


def save_seq_stats_report(
    stats_dict: dict,
    output_path: Path,
    mode: str = "fixed",
    window_size: int = 0,
    step_size: int = 0,
) -> None:
    """
    シーケンス長統計をtxtファイルに保存し、コンソールに表示する。
    
    Parameters
    ----------
    stats_dict : dict
        {ratio: {"train": stats, "test_normal": stats, "test_abnormal": stats}, ...}
    output_path : Path
        保存先のファイルパス
    mode : str
        sliding_windowのモード ("time" or "fixed")
    window_size : int
        ウィンドウサイズ
    step_size : int
        ステップサイズ
    """
    from datetime import datetime
    
    lines = []
    lines.append("=" * 70)
    lines.append("Sequence Length Statistics Report")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Mode: {mode}")
    lines.append(f"Window Size: {window_size}")
    lines.append(f"Step Size: {step_size}")
    lines.append("=" * 70)
    
    for ratio, ratio_stats in sorted(stats_dict.items()):
        lines.append(f"\n[Ratio: {ratio}]")
        lines.append("-" * 50)
        
        for data_type, stats in ratio_stats.items():
            lines.append(f"  {data_type}:")
            lines.append(f"    Count:    {stats['count']:,}")
            lines.append(f"    Avg Len:  {stats['avg_len']:.2f}")
            lines.append(f"    Min Len:  {stats['min_len']}")
            lines.append(f"    Max Len:  {stats['max_len']}")
            lines.append(f"    Std Dev:  {stats['std_len']:.2f}")
    
    lines.append("\n" + "=" * 70)
    
    report = "\n".join(lines)
    
    # コンソールに表示
    print(report)
    
    # ファイルに保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nStatistics saved to: {output_path}")



def prepare_npz_data(
    logdata_filepath: Path,
    output_dir: Path,
    window_size: int = 100,
    step_size: int = 50,
    mode: str = "fixed",
    features: list = ["EventId", "deltaT"]
) -> None:
    """
    CSVログデータから timeVAE 用の .npz データセットを作成する関数。
    prepare_model_data のロジックを踏襲しつつ、出力形式を npz に変更。
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    data = pd.read_csv(logdata_filepath)
    
    #----------- 諸操作 ----------#
    if "Label" not in data.columns:
        raise ValueError("Label column not found in CSV.")
    # Labelが文字("-")なら0(正常)、それ以外なら1(異常)とするロジックを踏襲
    data["Label"] = data["Label"].apply(lambda x: int(x != "-"))
    
    # タイムスタンプ計算
    if "TimeCreated_SystemTime" in data.columns:
        # format='mixed' is for pandas >= 2.0. For older versions, let it infer or specify.
        # Removing format='mixed' to be safe with pandas 1.3.4
        data["datetime"] = pd.to_datetime(data["TimeCreated_SystemTime"], errors="coerce")
        data["timestamp"] = data["datetime"].view("int64") // 10**9
        data["deltaT"] = data["datetime"].diff().dt.total_seconds().fillna(0)
    else:
        # 必須カラムがない場合のフォールバック（テスト用や別形式用）
        print("Warning: 'TimeCreated_SystemTime' not found. Ensure 'timestamp' and 'deltaT' exist or are not needed.")
    
    # カラム存在チェック
    required_cols = ["Label", "timestamp", "deltaT"] + [f for f in features if f not in ["deltaT"]]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        # EventIdなどは sliding_window で必須
        if "EventId" in missing and "EventId" in features:
             raise ValueError(f"Missing columns: {missing}")

    # ----------- データフレーム → モデル前データ ----------#
    # sliding_window は ["timestamp", "Label", "EventId", "deltaT"] を期待しているため、
    # それらが含まれているか確認してから渡す。
    # ここでは既存の sliding_window を再利用するため、必要なカラムを渡す。
    # もし features に EventId 以外が含まれる場合、sliding_window の改修が必要になる可能性があるが、
    # 現状の sliding_window は [ts, label, key, dt] をハードコードして処理している。
    # 汎用性を持たせるには sliding_window も改修すべきだが、
    # 今回は deeplog_file_generator の代替ということなので、EventId と deltaT を基本とする。
    
    cols_to_use = ["timestamp", "Label", "EventId", "deltaT"]
    deeplog_df = sliding_window(
        data[cols_to_use],
        para={"window_size": window_size, "step_size": step_size},
        mode = mode
    )
    
    # ---------------------------------------------------------
    # シーケンス長の一貫性チェックとパディング処理
    # ---------------------------------------------------------
    # すべてのfeature列で長さが一致しているか、また行ごとの長さが一定かを確認
    # もし一致していなければパディングを行う
    
    # まず基準となる feature (features[0]) の長さを確認
    base_feature = features[0]
    lengths = deeplog_df[base_feature].apply(len)
    max_len = lengths.max()
    min_len = lengths.min()
    
    if max_len != min_len:
        print(f"Warning: Sequence lengths are inconsistent (min={min_len}, max={max_len}). Scaling to max_len={max_len} with padding.")
        
        # パディング関数 (後ろ埋め, 値は 0)
        def pad_seq(seq, target_len, pad_val=0):
            seq_len = len(seq)
            if seq_len < target_len:
                # numpy arrayの場合と listの場合で処理を分ける
                if isinstance(seq, np.ndarray):
                    # shape (L,) -> (Target,)
                    return np.pad(seq, (0, target_len - seq_len), 'constant', constant_values=pad_val)
                else:
                    return list(seq) + [pad_val] * (target_len - seq_len)
            return seq
            
        # 必要なカラムすべてに対してパディング適用
        for feat in features:
            deeplog_df[feat] = deeplog_df[feat].apply(lambda x: pad_seq(x, max_len, 0))
            
    else:
        print(f"Sequence lengths are consistent (len={max_len}). No padding needed.")

    
    # normalとabnormalを切り分け
    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]

    # shuffle
    df_normal = df_normal.sample(frac=1, random_state=42).reset_index(drop=True)  
    normal_len = len(df_normal)
    
    # データを保存
    # prepare_model_data では train_ratio ループがあるが、
    # ここではシンプルに 0.8 (標準的) 固定、あるいはループさせるか。
    # timeVAEのパイプラインは単一のデータセットを期待するため、
    # 複数の ratio で生成すると管理が煩雑になる。
    # 一旦、ディレクトリを分けて保存する形式を採用する。

    train_ratio_list = [0.8] # デフォルトで0.8のみ作成、必要ならリストを引数化
    
    for train_ratio in train_ratio_list:
        train_len = int(normal_len * train_ratio)
        
        # 保存先ディレクトリ (例: output_dir/ratio_0.8/)
        save_dir = output_dir / f'ratio_{train_ratio}'
        os.makedirs(save_dir, exist_ok=True)

        # train
        train = df_normal[:train_len]
        npz_file_generator(
            filename = str(save_dir / 'train.npz'),
            df = train,
            features = features,
        )
        print(f"[{train_ratio}] training size {train_len} -> {save_dir / 'train.npz'}")

        # test(normal)
        test_normal = df_normal[train_len:]
        npz_file_generator(
            filename = str(save_dir / 'test_normal.npz'),
            df = test_normal,
            features = features,
        )
        print(f"[{train_ratio}] test normal size {normal_len - train_len} -> {save_dir / 'test_normal.npz'}")

        # abnormal
        npz_file_generator(
            filename = str(save_dir / 'test_abnormal.npz'),
            df = df_abnormal,
            features = features, 
        )
        print(f"[{train_ratio}] test abnormal size {len(df_abnormal)} -> {save_dir / 'test_abnormal.npz'}")

def prepare_integrated_model_data(
    logdata_filepath:Path,
    output_dir:Path,
    project_list:list,
    window_size:int = 300,
    step_size:int = 60,
    mode: str = "time", 
) -> None:
    """
    統合データ用。
    モデル前データ作成工程の親関数。
    vocabファイル作成まで行う。
    """
    output_dir.mkdir(exist_ok=True)
    
    data = pd.read_csv(logdata_filepath)
    
    #----------- 諸操作 ----------#
    if "Label" not in data.columns:
        raise ValueError("Label column not found in CSV.")
    data["Label"] = data["Label"].apply(lambda x: int(x != "-"))
    data["datetime"] = pd.to_datetime(data["TimeCreated_SystemTime"], format='mixed')
    data["timestamp"] = data["datetime"].view("int64") // 10**9  
    data["deltaT"] = data["datetime"].diff().dt.total_seconds().fillna(0)
    
    # ----------- データフレーム → モデル前データ ----------#
    
    # ratio = 1.0 はvocab作成用
    train_ratio_list = [0.6, 0.8, 1.0]
    
    # シーケンス長統計を格納する辞書
    seq_stats = {}

    for train_ratio in train_ratio_list:
        integrated_train = pd.DataFrame()
        df_normal = pd.DataFrame()
        
        # このratioの統計を格納
        ratio_stats = {"train": None, "test_normal_all": [], "test_abnormal_all": []}
        
        # projectごとに処理
        for project in project_list:
            # プロジェクトごとにデータをフィルタリング
            project_data = data[data["project"] == project]

            # sampling with sliding window
            deeplog_df = sliding_window(
                project_data[["timestamp", "Label", "EventId", "deltaT"]],
                para={"window_size": window_size, "step_size": step_size},
                mode=mode,
            )
            deeplog_df["project"] = project

            # 余事象データは即ち正常データなので、即座に統合
            if project.endswith("_C"):
                integrated_train = pd.concat([integrated_train, deeplog_df], ignore_index=True)
                continue

            # normalとabnormalを切り分け
            temp_normal = deeplog_df[deeplog_df["Label"] == 0]
            temp_abnormal = deeplog_df[deeplog_df["Label"] == 1]

            if(train_ratio == 1.0):
                df_normal = pd.concat([df_normal, temp_normal], ignore_index=True)
                continue

            save_dir = output_dir/f"ratio_{str(train_ratio)}"/project
            os.makedirs(save_dir, exist_ok=True)
        
            # shuffle
            temp_normal = temp_normal.sample(frac=1, random_state=12).reset_index(drop=True)  
            temp_abnormal = temp_abnormal.sample(frac=1, random_state=12).reset_index(drop=True)  
            normal_len = len(temp_normal)

            train_len = int(normal_len * train_ratio)

            # train
            train = temp_normal[:train_len]
            integrated_train = pd.concat([integrated_train, train], ignore_index=True)

            # test(normal)
            test_normal = temp_normal[train_len:]
            deeplog_file_generator(
                filename = str(save_dir) + '/test_normal',
                df = test_normal,
                features = ["EventId", "deltaT"], 
            )
            
            # test_normalの統計を収集
            ratio_stats["test_normal_all"].append(test_normal)

            # test(abnormal)
            test_abnormal = temp_abnormal
            deeplog_file_generator(
                filename = str(save_dir) + '/test_abnormal',
                df = test_abnormal,
                features = ["EventId", "deltaT"], 
            )
            
            # test_abnormalの統計を収集
            ratio_stats["test_abnormal_all"].append(test_abnormal)
            
        if(train_ratio == 1.0):
            continue

        save_dir = output_dir/f"ratio_{str(train_ratio)}"
        deeplog_file_generator(
            filename = str(save_dir) + '/train',
            df = integrated_train,
            features = ["EventId", "deltaT"], 
        )
        
        # このratioの統計を計算
        test_normal_combined = pd.concat(ratio_stats["test_normal_all"], ignore_index=True) if ratio_stats["test_normal_all"] else pd.DataFrame()
        test_abnormal_combined = pd.concat(ratio_stats["test_abnormal_all"], ignore_index=True) if ratio_stats["test_abnormal_all"] else pd.DataFrame()
        
        seq_stats[train_ratio] = {
            "train": calculate_seq_length_stats(integrated_train, "EventId"),
            "test_normal": calculate_seq_length_stats(test_normal_combined, "EventId"),
            "test_abnormal": calculate_seq_length_stats(test_abnormal_combined, "EventId"),
        }
        
    # vocab 作成
    save_dir = output_dir/'vocab'
    os.makedirs(save_dir, exist_ok=True)

    deeplog_file_generator(
        filename = str(save_dir) + '/train',
        df = df_normal,
        features = ["EventId"], # EventId only
    )
    print("vocab size {}".format(len(df_normal)))
    
    # シーケンス長統計をファイルに保存
    stats_output_path = output_dir / "seq_stats.txt"
    save_seq_stats_report(
        stats_dict=seq_stats,
        output_path=stats_output_path,
        mode=mode,
        window_size=window_size,
        step_size=step_size,
    )
    
    return
    

# ---------------------------------------------------- 余談部分 -----------------------------------------------------#
def _stratified_sample_one(
    df: pd.DataFrame,
    target_n: int,
    event_id_col: str = "EventID",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    単一の DataFrame から、EventID ごとに層別抽出して target_n 件サンプリングする。
    すべての EventID から最低 1 件は含めることを保証する版。
    """
    df = df.dropna(subset=[event_id_col]).copy()
    df[event_id_col] = df[event_id_col].astype(str)

    counts = df[event_id_col].value_counts()
    event_ids = counts.index
    K = len(event_ids)

    if target_n < K:
        raise ValueError(
            f"target_n={target_n} ではユニーク EventID 数 {K} に対して "
            "各 EventID 最低1件を保証できません。"
        )

    # ① まず各 EventID に 1 件ずつ割り当て（最低保証）
    alloc = pd.Series(1, index=event_ids, dtype=int)

    # 残りを出現頻度に応じて配分
    remaining = target_n - K
    if remaining > 0:
        extra = (counts / counts.sum() * remaining).round().astype(int)
        alloc += extra

        # 合計が target_n からずれたら微調整（>=1 を維持しつつ）
        diff = target_n - alloc.sum()
        if diff != 0:
            # 出現数の多い順
            sorted_ids = counts.index
            step = 1 if diff > 0 else -1
            diff_abs = abs(diff)
            i = 0
            while diff_abs > 0:
                eid = sorted_ids[i % len(sorted_ids)]
                # 減らす場合は 1 未満にはしない
                if step < 0 and alloc[eid] <= 1:
                    i += 1
                    continue
                alloc[eid] += step
                diff_abs -= 1
                i += 1

    # ② 実際にサンプリング
    sampled_list = []
    for event_id, n in alloc.items():
        group = df[df[event_id_col] == event_id]
        n_actual = min(n, len(group))
        if n_actual <= 0:
            continue
        # group が少なすぎて n_actual < 1 になりうるケースへの対処を入れるならここ
        sampled = group.sample(n=n_actual, random_state=random_state)
        sampled_list.append(sampled)

    sampled_df = pd.concat(sampled_list, ignore_index=False)

    # ここでは基本的に len(sampled_df) == target_n になる想定。
    # もし多少ずれるのを許容するなら、そのままでもOK。
    return sampled_df


def stratified_sample_by_eventid_two_sets(
    input_files: List[Union[str, Path]],
    output_file1: Union[str, Path],
    output_file2: Union[str, Path],
    target_n_each: int = 5000,
    event_id_col: str = "EventID",
    random_state: int = 42,
):
    """
    複数のイベントログ CSV を結合し、EventID ごとの層別抽出で
    ・サンプル1: target_n_each 件
    ・サンプル2: target_n_each 件
    の2セットを重複なしで作成し、別々の CSV に保存する。

    Parameters
    ----------
    input_files : list of str or Path
        結合する元 CSV ファイルのリスト
    output_file1 : str or Path
        1つ目のサンプルの保存パス
    output_file2 : str or Path
        2つ目のサンプルの保存パス
    target_n_each : int, default 5000
        各サンプルに含めたい件数
    event_id_col : str, default "EventID"
        EventID の列名
    random_state : int, default 42
        ランダムシード（2つ目は random_state+1 を使う）
    """

    paths = [Path(f) for f in input_files]
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    print(f"結合後の全レコード数: {len(df)}")

    # 2セット分の件数があるかチェック
    if len(df) < 2 * target_n_each:
        raise ValueError(
            f"データ数が不足しています: レコード数 {len(df)} に対して "
            f"2×{target_n_each} 件は確保できません。"
        )

    # ---- サンプル1 ----
    sampled1 = _stratified_sample_one(
        df=df,
        target_n=target_n_each,
        event_id_col=event_id_col,
        random_state=random_state,
    )
    print(f"サンプル1件数: {len(sampled1)}")

    # サンプル1を元データから除外（index ベースで削除）
    remaining_df = df.drop(index=sampled1.index)

    print(f"サンプル1除外後の残りレコード数: {len(remaining_df)}")

    # ---- サンプル2 ----
    sampled2 = _stratified_sample_one(
        df=remaining_df,
        target_n=target_n_each,
        event_id_col=event_id_col,
        random_state=random_state + 1,
    )
    print(f"サンプル2件数: {len(sampled2)}")

    # 念のためサンプル間の重複確認
    overlap = set(sampled1.index) & set(sampled2.index)
    print(f"サンプル1・2の重複インデックス数: {len(overlap)}")

    # ---- 保存 ----
    output_path1 = Path(output_file1)
    output_path2 = Path(output_file2)

    sampled1.to_csv(output_path1, index=False)
    sampled2.to_csv(output_path2, index=False)

    print(f"サンプル1を保存しました: {output_path1}")
    print(f"サンプル2を保存しました: {output_path2}")

    return sampled1, sampled2

def delete_unwanted_logs(
    input_filepath:Path,
    start_date:str,
    end_date:str,
    output_filepath:Path = None,
) -> pd.DataFrame:
    """
    指定された日付範囲のログを削除する。
    
    Parameters
    ----------
    input_filepath : Path
        元のログファイルのパス
    start_date : str
        開始日付（YYYY-MM-DD）
    end_date : str
        終了日付（YYYY-MM-DD）
    output_filepath : Path, optional
        出力ファイルのパス（デフォルト: 元のファイル）
        
    Returns
    -------
    pd.DataFrame
        削除後のデータフレーム
    """
    data = pd.read_csv(input_filepath)

    data["TimeCreated_SystemTime"] = pd.to_datetime(
        data["TimeCreated_SystemTime"], 
        format='mixed',      
    )
    data["date"] = data["TimeCreated_SystemTime"].dt.date

    filtered = data[
        (data["date"] >= pd.to_datetime(start_date).date()) &
        (data["date"] <= pd.to_datetime(end_date).date())
    ]
    # 出力ファイルが指定されていない場合、元のファイルに上書き保存
    if output_filepath is None:
        output_filepath = input_filepath

    # 結果を保存
    filtered.to_csv(output_filepath, index=False)
    return filtered
    

def calculate_equivalent_window_params(
    csv_filepath: Union[str, Path],
    source_mode: str,  # "time" or "fixed"
    source_window_size: float,
    source_step_size: float,
    target_mode: str,  # "time" or "fixed"
) -> dict:
    """
    CSVファイルを読み込み、指定されたwindow_sizeおよびstep_sizeに対して、
    平均シーケンス長を合わせるために、もう一方の手法で必要なパラメータを算出する。

    Parameters
    ----------
    csv_filepath : Union[str, Path]
        入力CSVファイルのパス。
        必須カラム: TimeCreated_SystemTime（タイムスタンプ）
        ※ T1105/security2.csv の形式を想定

    source_mode : str
        変換元の方式。"time" または "fixed"

    source_window_size : float
        変換元のwindow_size。
        - timeモード: 秒単位の時間幅
        - fixedモード: イベント数

    source_step_size : float
        変換元のstep_size。
        - timeモード: 秒単位の時間幅
        - fixedモード: イベント数

    target_mode : str
        変換先の方式。"time" または "fixed"

    Returns
    -------
    dict
        {
            "window_size": float or int,  # 変換先のwindow_size
            "step_size": float or int,    # 変換先のstep_size
            "avg_event_rate": float,      # 平均イベント発生率 (events/second)
            "avg_time_per_event": float,  # 平均イベント間隔 (seconds/event)
            "total_events": int,          # 総イベント数
            "total_duration": float,      # 総時間 (seconds)
            "source_avg_sequence_length": float,  # 変換元の平均シーケンス長
        }

    Raises
    ------
    ValueError
        source_mode と target_mode が同じ場合、または無効な値の場合
    FileNotFoundError
        CSVファイルが見つからない場合
    """
    # モードの検証
    if source_mode == target_mode:
        raise ValueError("source_mode と target_mode は異なる値を指定してください。")
    
    if source_mode not in ("time", "fixed"):
        raise ValueError("source_mode は 'time' または 'fixed' を指定してください。")
    
    if target_mode not in ("time", "fixed"):
        raise ValueError("target_mode は 'time' または 'fixed' を指定してください。")

    # CSVファイルの読み込み
    csv_filepath = Path(csv_filepath)
    if not csv_filepath.exists():
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_filepath}")
    
    data = pd.read_csv(csv_filepath)
    
    if data.shape[0] == 0:
        raise ValueError("CSVファイルにデータがありません。")
    
    # 必須カラムの確認
    if "TimeCreated_SystemTime" not in data.columns:
        raise ValueError("TimeCreated_SystemTime カラムが見つかりません。")
    
    # タイムスタンプを変換してUNIX秒に
    data["datetime"] = pd.to_datetime(data["TimeCreated_SystemTime"], format='mixed')
    data["timestamp"] = data["datetime"].view("int64") // 10**9
    
    # 統計値の計算
    time_data = data["timestamp"]
    total_events = len(time_data)
    total_duration = float(time_data.max() - time_data.min())
    
    if total_duration <= 0:
        raise ValueError("データの時間範囲が0秒以下です。時間ベースの変換ができません。")
    
    # 平均イベント率 (events/second)
    avg_event_rate = total_events / total_duration
    # 平均イベント間隔 (seconds/event)
    avg_time_per_event = total_duration / total_events

    result = {
        "csv_filepath": str(csv_filepath),
        "avg_event_rate": avg_event_rate,
        "avg_time_per_event": avg_time_per_event,
        "total_events": total_events,
        "total_duration": total_duration,
    }

    if source_mode == "time" and target_mode == "fixed":
        # time → fixed への変換
        # timeモードでのwindow_size秒間に含まれる平均イベント数 = window_size * avg_event_rate
        source_avg_seq_len = source_window_size * avg_event_rate
        target_window_size = int(round(source_avg_seq_len))
        
        # step_sizeも同様に変換
        source_step_events = source_step_size * avg_event_rate
        target_step_size = int(round(source_step_events))
        
        # 最小値を1に制限
        target_window_size = max(1, target_window_size)
        target_step_size = max(1, target_step_size)
        
        result["window_size"] = target_window_size
        result["step_size"] = target_step_size
        result["source_avg_sequence_length"] = source_avg_seq_len
        
    elif source_mode == "fixed" and target_mode == "time":
        # fixed → time への変換
        # fixedモードでのwindow_sizeイベントに相当する時間 = window_size * avg_time_per_event
        source_avg_seq_len = source_window_size  # fixedでは固定
        target_window_size = source_window_size * avg_time_per_event
        
        # step_sizeも同様に変換
        target_step_size = source_step_size * avg_time_per_event
        
        result["window_size"] = target_window_size
        result["step_size"] = target_step_size
        result["source_avg_sequence_length"] = source_avg_seq_len

    return result


def print_equivalent_params_summary(params: dict, source_mode: str, target_mode: str) -> None:
    """
    calculate_equivalent_window_params の結果を整形して表示するユーティリティ関数。

    Parameters
    ----------
    params : dict
        calculate_equivalent_window_params の戻り値
    source_mode : str
        変換元モード
    target_mode : str
        変換先モード
    """
    print("=" * 60)
    print("Sliding Window パラメータ変換結果")
    print("=" * 60)
    print(f"入力CSV: {params.get('csv_filepath', 'N/A')}")
    print(f"変換方向: {source_mode} → {target_mode}")
    print("-" * 60)
    print("【データ統計】")
    print(f"  総イベント数:    {params['total_events']:,}")
    print(f"  総時間:          {params['total_duration']:.2f} 秒")
    print(f"  平均イベント率:  {params['avg_event_rate']:.4f} events/sec")
    print(f"  平均イベント間隔: {params['avg_time_per_event']:.4f} sec/event")
    print("-" * 60)
    print("【変換結果】")
    print(f"  推奨 window_size: {params['window_size']}")
    print(f"  推奨 step_size:   {params['step_size']}")
    print(f"  元の平均シーケンス長: {params['source_avg_sequence_length']:.2f}")
    print("=" * 60)
