from pathlib import Path
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import csv
from typing import List, Union

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
        例）columns = [timestamp, Label, EventID, EventId, deltaT] など
        少なくとも ["timestamp", "Label", "deltaT"] を含むこと。

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
        各セルには配列（timestamp列は時刻配列、Labelはウィンドウラベル、など）が入る。
    """
    if raw_data.shape[0] == 0:
        raise ValueError("raw_data is empty")

    # 必須列チェック
    required_cols = ["timestamp", "Label", "deltaT"]
    for c in required_cols:
        if c not in raw_data.columns:
            raise ValueError(f"required column '{c}' not found in raw_data")

    # 共通で使う列
    time_data   = raw_data["timestamp"]
    label_data  = raw_data["Label"]
    deltaT_data = raw_data["deltaT"]

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
            ts_seq    = time_data.iloc[start_index:end_index].values
            label_seq = label_data.iloc[start_index:end_index].values
            dt_seq    = deltaT_data.iloc[start_index:end_index].values.copy()
            dt_seq[0] = 0  # 先頭は0
            window_label = label_seq.max()

            # 列順 raw_data.columns に合わせて1行分を組み立てる
            row = []
            for col in raw_data.columns:
                if col == "timestamp":
                    row.append(ts_seq)
                elif col == "Label":
                    row.append(window_label)
                elif col == "deltaT":
                    row.append(dt_seq)
                else:
                    # EventID / EventId / project など、それ以外の列はそのままシーケンス化
                    row.append(raw_data[col].iloc[start_index:end_index].values)

            new_data.append(row)

        print("there are %d instances (sliding windows) in this dataset\n" % len(new_data))
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

            ts_seq    = time_data.iloc[start_index:end_index].values
            label_seq = label_data.iloc[start_index:end_index].values
            dt_seq    = deltaT_data.iloc[start_index:end_index].values.copy()
            dt_seq[0] = 0
            window_label = label_seq.max()

            row = []
            for col in raw_data.columns:
                if col == "timestamp":
                    row.append(ts_seq)
                elif col == "Label":
                    row.append(window_label)
                elif col == "deltaT":
                    row.append(dt_seq)
                else:
                    row.append(raw_data[col].iloc[start_index:end_index].values)

            new_data.append(row)

            num_session += 1
            if num_session % 1000 == 0:
                print(f"process {num_session} count window", end="\r")

        print("there are %d instances (sliding windows) in this dataset\n" % num_session)
        return pd.DataFrame(new_data, columns=raw_data.columns)

    else:
        raise ValueError('mode must be either "time" or "fixed".')


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


def deeplog_file_generator(
    filename, 
    df, 
    features
) -> None:
    """
    データフレームを deeplog_file に変換して保存する関数。
    """
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(",".join([str(v) for v in val]) + " ")
            f.write("\n")
    

def prepare_deeplog_file(
    logdata_filepath:Path,
    output_dir:Path,
    features:List[str] = ["EventID"],
    window_size:int = 100,
    step_size:int = 50,
    mode: str = "fixed", 
) -> None:
    """
    モデル前データ作成工程の親関数。
    vocabファイル作成まで行う。
    アノテーション済みのcsvを入力に想定。
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    data = pd.read_csv(logdata_filepath)
    
    #----------- 諸操作 ----------#
    if "Label" not in data.columns:
        raise ValueError("Label column not found in CSV.")
    data["Label"] = data["Label"].apply(lambda x: int(x != "-"))
    data["datetime"] = pd.to_datetime(data["TimeCreated_SystemTime"], errors="coerce")
    data["timestamp"] = data["datetime"].view("int64") // 10**9  
    data["deltaT"] = data["datetime"].diff().dt.total_seconds().fillna(0)
    
    # ----------- データフレーム → モデル前データ ----------#
    # sampling with sliding window
    deeplog_df = sliding_window(
        data[["timestamp", "Label", "EventID", "EventId", "deltaT"]],
        #para={"window_size": int(window_size) * 60, "step_size": int(step_size) * 60},
        para={"window_size": window_size, "step_size": step_size},
        mode = mode
    )
    
    # normalとabnormalを切り分け
    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]

    # shuffle
    df_normal = df_normal.sample(frac=1, random_state=42).reset_index(drop=True)  
    normal_len = len(df_normal)
    
    # シーケンス長統計を格納する辞書
    seq_stats = {}
    
    train_ratio_list = [0.8]
    for train_ratio in train_ratio_list:

        train_len = int(normal_len * train_ratio)
        save_dir = output_dir/f'ratio_{train_ratio}'

        os.makedirs(save_dir, exist_ok=True)

        # train
        train = df_normal[:train_len]
        deeplog_file_generator(
            filename = str(save_dir) + '/train',
            df = train,
            features = features,
        )
        print("training size {}".format(train_len))

        # test(normal)
        test_normal = df_normal[train_len:]
        deeplog_file_generator(
            filename = str(save_dir) + '/test_normal',
            df = test_normal,
            features = features,
        )
        print("test normal size {}".format(normal_len - train_len))

        deeplog_file_generator(
            filename = str(save_dir) + '/test_abnormal',
            df = df_abnormal,
            features = features, 
        )
        print("test abnormal size {}".format(len(df_abnormal)))
        
        # シーケンス長統計を計算
        seq_stats[train_ratio] = {
            "train": calculate_seq_length_stats(train, "EventId"),
            "test_normal": calculate_seq_length_stats(test_normal, "EventId"),
            "test_abnormal": calculate_seq_length_stats(df_abnormal, "EventId"),
        }
        
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


def prepare_integrated_deeplog_file(
    logdata_filepath:Path,
    output_dir:Path = INTERIM_DIR/"Integrated",
    features:List[str] = ["EventID"],
    window_size:int = 300,
    step_size:int = 60,
    mode: str = "time", 
) -> None:
    """
    統合データ用。
    モデル前データ作成工程の親関数。
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    data = pd.read_csv(logdata_filepath)
    
    #----------- 諸操作 ----------#
    if "Label" not in data.columns:
        raise ValueError("Label column not found in CSV.")
    data["Label"] = data["Label"].apply(lambda x: int(x != "-"))
    data["datetime"] = pd.to_datetime(data["TimeCreated_SystemTime"], errors="coerce")
    data["timestamp"] = data["datetime"].view("int64") // 10**9  
    data["deltaT"] = data["datetime"].diff().dt.total_seconds().fillna(0)
    
    # ----------- データフレーム → モデル前データ ----------#
    
    project_list = data["project"].unique()
    
    train_ratio_list = [0.8]
    
    # シーケンス長統計を格納する辞書
    seq_stats = {}

    for train_ratio in train_ratio_list:
        integrated_train = pd.DataFrame()
        
        # このratioの統計を格納
        ratio_stats = {"train": None, "test_normal_all": [], "test_abnormal_all": []}
        
        # projectごとに処理
        for project in project_list:
            # プロジェクトごとにデータをフィルタリング
            project_data = data[data["project"] == project]

            # sampling with sliding window
            deeplog_df = sliding_window(
                project_data[["timestamp", "Label", "EventID","EventId", "deltaT"]],
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
                features = features, 
            )
            
            # test_normalの統計を収集
            ratio_stats["test_normal_all"].append(test_normal)

            # test(abnormal)
            test_abnormal = temp_abnormal
            deeplog_file_generator(
                filename = str(save_dir) + '/test_abnormal',
                df = test_abnormal,
                features = features, 
            )
            
            # test_abnormalの統計を収集
            ratio_stats["test_abnormal_all"].append(test_abnormal)
            
        # concat した train
        save_dir = output_dir/f"ratio_{str(train_ratio)}"
        deeplog_file_generator(
            filename = str(save_dir) + '/train',
            df = integrated_train,
            features = features, 
        )
        
        # このratioの統計を計算
        test_normal_combined = pd.concat(ratio_stats["test_normal_all"], ignore_index=True) if ratio_stats["test_normal_all"] else pd.DataFrame()
        test_abnormal_combined = pd.concat(ratio_stats["test_abnormal_all"], ignore_index=True) if ratio_stats["test_abnormal_all"] else pd.DataFrame()
        
        seq_stats[train_ratio] = {
            "train": calculate_seq_length_stats(integrated_train, "EventId"),
            "test_normal": calculate_seq_length_stats(test_normal_combined, "EventId"),
            "test_abnormal": calculate_seq_length_stats(test_abnormal_combined, "EventId"),
        }
        
    
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


def convert_deeplog_to_ohe_npz(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> None:
    """
    指定ディレクトリ内の DeepLog 形式のファイルを読み込み、One-Hot Encoding を適用して .npz に保存する。
    
    Parameters
    ----------
    input_dir : str or Path
        入力ディレクトリ (例: data/interim/T1105/ratio_0.8)
        train, test_normal, test_abnormal があることを想定
    output_dir : str or Path
        出力ディレクトリ (例: data/processed/T1105/ratio_0.8)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ボキャブラリー作成のため train, test_normal, test_abnormal 全てを読み込む
    target_files = ["train", "test_normal", "test_abnormal"]
    vocab = set()
    
    print("Building vocabulary from all data files...")
    
    for filename in target_files:
        fpath = input_dir / filename
        if not fpath.exists():
            continue
            
        with open(fpath, "r") as f:
            for line in f:
                tokens = line.strip().split()
                for t in tokens:
                    vocab.add(t)

    sorted_vocab = sorted(list(vocab))
    token_to_id = {token: i for i, token in enumerate(sorted_vocab)}
    vocab_size = len(sorted_vocab)
    print(f"Vocab size: {vocab_size}")
    
    # 配列を処理する内部関数
    def process_file(filename: Path, save_name: str):
        if not filename.exists():
            print(f"Warning: {filename} does not exist. Skipping.")
            return

        print(f"Processing {filename} ...")
        
        data_list = []
        max_seq_len = 0
        
        # まず読み込み
        lines_data = []
        with open(filename, "r") as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                lines_data.append(tokens)
                max_seq_len = max(max_seq_len, len(tokens))
        
        print(f"  Max sequence length: {max_seq_len}")
        
        # OHE 変換
        # Shape: (N, T, V)
        N = len(lines_data)
        T = max_seq_len
        V = vocab_size
        
        data_np = np.zeros((N, T, V), dtype=np.float32)
        
        for i, tokens in enumerate(lines_data):
            for t, token in enumerate(tokens):
                if token in token_to_id:
                    idx = token_to_id[token]
                    data_np[i, t, idx] = 1.0
                else:
                    # OOV は発生しないはずだが念のため
                    pass 
                    
        # 保存
        out_path = output_dir / save_name
        np.savez(out_path, data=data_np)
        print(f"  Saved to {out_path} (Shape: {data_np.shape})")

    # 各ファイルを処理
    for fname in target_files:
        process_file(input_dir / fname, f"{fname}.npz")
    
    # Vocabも保存しておく
    import json
    with open(output_dir / "vocab.json", "w") as f:
        json.dump(token_to_id, f, indent=4)
        
    print("Done.")


def convert_deeplog_to_embedding_npz(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    embedding_dir: Union[str, Path],
) -> None:
    """
    指定ディレクトリ内の DeepLog 形式のファイルを読み込み、埋め込みベクトルに変換して .npz に保存する。
    
    Parameters
    ----------
    input_dir : str or Path
        入力ディレクトリ (例: data/interim/T1105(full)/ratio_0.8)
    output_dir : str or Path
        出力ディレクトリ (例: data/processed/T1105(full)/ratio_0.8)
    embedding_dir : str or Path
        埋め込みファイルがあるディレクトリ (例: data/embedding)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    embedding_dir = Path(embedding_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 埋め込みデータのロード
    json_path = embedding_dir / "eventid_sentence_bert_embeddings.json"
    npy_path = embedding_dir / "eventid_sentence_bert_embeddings.npy"
    
    if not json_path.exists() or not npy_path.exists():
        raise FileNotFoundError(f"Embedding files not found in {embedding_dir}")
        
    print(f"Loading embeddings from {embedding_dir} ...")
    import json
    with open(json_path, "r") as f:
        meta = json.load(f)
        event_ids_in_emb = meta["event_ids"] # list of str
        
    embeddings_matrix = np.load(npy_path)
    # event_ids_in_emb[i] corresponds to embeddings_matrix[i]
    
    # EventID -> Index のマップ作成
    # 読み込むデータの event_id は文字列として扱う
    eid_to_index = {eid: i for i, eid in enumerate(event_ids_in_emb)}
    vocab_size, embedding_dim = embeddings_matrix.shape
    print(f"Embedding Vocab Size: {vocab_size}, Dimension: {embedding_dim}")

    target_files = ["train", "test_normal", "test_abnormal"]
    
    def process_file(filename: Path, save_name: str):
        if not filename.exists():
            print(f"Warning: {filename} does not exist. Skipping.")
            return

        print(f"Processing {filename} ...")
        
        # まず読み込み
        lines_data = []
        max_seq_len = 0
        with open(filename, "r") as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                lines_data.append(tokens)
                max_seq_len = max(max_seq_len, len(tokens))
        
        print(f"  Max sequence length: {max_seq_len}")
        
        # Embedding 変換
        # Shape: (N, T, E)
        N = len(lines_data)
        T = max_seq_len
        E = embedding_dim
        
        data_np = np.zeros((N, T, E), dtype=np.float32)
        
        oov_count = 0
        total_tokens = 0
        oov_tokens = set()
        
        for i, tokens in enumerate(lines_data):
            for t, token in enumerate(tokens):
                total_tokens += 1
                if token in eid_to_index:
                    idx = eid_to_index[token]
                    data_np[i, t, :] = embeddings_matrix[idx]
                else:
                    # OOV: Zero vector is already set by initialization, just count it
                    oov_count += 1
                    oov_tokens.add(token)
                    
        if oov_count > 0:
            print(f"  Warning: {oov_count}/{total_tokens} tokens were OOV (initialized to zero vectors).")
            print(f"  OOV Tokens: {sorted(list(oov_tokens))}")
        else:
            print("  No OOV tokens found.")
                    
        # 保存
        out_path = output_dir / save_name
        np.savez(out_path, data=data_np)
        print(f"  Saved to {out_path} (Shape: {data_np.shape})")

    # 各ファイルを処理
    for fname in target_files:
        process_file(input_dir / fname, f"{fname}.npz")
        
    print("Done.")


if __name__ == "__main__":
    # Example usage (OHE)
    inp = Path("../data/interim/T1105/ratio_0.8")
    out = Path("../data/processed/T1105/ratio_0.8")
    if inp.exists():
        convert_deeplog_to_ohe_npz(inp, out)

    # Example usage (Embedding)
    inp_emb = Path("../data/interim/T1105(full)/ratio_0.8")
    out_emb = Path("../data/processed/T1105(full)/ratio_0.8")
    emb_dir = Path("../data/embedding")
    
    if inp_emb.exists() and emb_dir.exists():
        print("\n--- Running Embedding Conversion ---")
        convert_deeplog_to_embedding_npz(inp_emb, out_emb, emb_dir)
