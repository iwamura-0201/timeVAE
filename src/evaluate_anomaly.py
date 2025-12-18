
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to python path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import load_scaler
from src.vae.vae_utils import load_vae_model

def evaluate_anomaly_detection(
    model_dir: str,
    test_data_dir: str,
    vae_type: str = None,  # Changed to None for auto-detection
    save_results_dir: str = None,
):
    """
    学習済みVAEモデルを用いて異常検知の評価を行います。
    
    Args:
        model_dir: 学習済みモデルとスケーラーが含まれるディレクトリ。
        test_data_dir: テストデータ (test_normal.npz, test_abnormal.npz) が含まれるディレクトリ。
        vae_type: VAEモデルの種類 (timeVAE, vae_dense, vae_conv, timeVAE_torch)。
                  Noneの場合、model_dirから自動検出します。
        save_results_dir: 評価結果やプロットを保存するディレクトリ。
    """
    if save_results_dir:
        os.makedirs(save_results_dir, exist_ok=True)

    # Auto-detect vae_type if not provided
    if vae_type is None:
        print(f"vae_typeが指定されていません。{model_dir}からモデルタイプを自動検出します...")
        
        # Check for PyTorch model files
        if os.path.exists(os.path.join(model_dir, "TimeVAE_weights.pth")):
            vae_type = "timeVAE_torch"
            print(f"  → PyTorchモデルを検出: vae_type='timeVAE_torch'")
        # Check for Keras/TensorFlow TimeVAE model files
        elif os.path.exists(os.path.join(model_dir, "TimeVAE_encoder_wts.h5")):
            vae_type = "timeVAE"
            print(f"  → Keras TimeVAEモデルを検出: vae_type='timeVAE'")
        # Check for dense VAE
        elif os.path.exists(os.path.join(model_dir, "VAE_Dense_encoder_wts.h5")):
            vae_type = "vae_dense"
            print(f"  → Dense VAEモデルを検出: vae_type='vae_dense'")
        # Check for conv VAE
        elif os.path.exists(os.path.join(model_dir, "VAE_Conv_encoder_wts.h5")):
            vae_type = "vae_conv"
            print(f"  → Conv VAEモデルを検出: vae_type='vae_conv'")
        else:
            raise FileNotFoundError(
                f"モデルファイルが見つかりません: {model_dir}\n"
                f"TimeVAE_weights.pth または TimeVAE_encoder_wts.h5 などのファイルが必要です。"
            )

    print(f"スケーラーを読み込んでいます: {model_dir}...")
    scaler = load_scaler(model_dir)

    print(f"モデル ({vae_type}) を読み込んでいます: {model_dir}...")
    vae = load_vae_model(vae_type, model_dir)

    # テストデータの読み込み
    normal_path = os.path.join(test_data_dir, "test_normal.npz")
    abnormal_path = os.path.join(test_data_dir, "test_abnormal.npz")

    if not os.path.exists(normal_path):
        raise FileNotFoundError(f"正常テストデータが見つかりません: {normal_path}")
    if not os.path.exists(abnormal_path):
        raise FileNotFoundError(f"異常テストデータが見つかりません: {abnormal_path}")

    print("テストデータを読み込んでいます...")
    test_normal = np.load(normal_path)["data"]
    test_abnormal = np.load(abnormal_path)["data"]
    
    print(f"正常データ形状: {test_normal.shape}")
    print(f"異常データ形状: {test_abnormal.shape}")

    # データのスケーリング
    scaled_normal = scaler.transform(test_normal)
    scaled_abnormal = scaler.transform(test_abnormal)

    # ラベルの作成 (0: 正常, 1: 異常)
    y_true_normal = np.zeros(len(scaled_normal))
    y_true_abnormal = np.ones(len(scaled_abnormal))
    
    X_test = np.concatenate([scaled_normal, scaled_abnormal], axis=0)
    y_true = np.concatenate([y_true_normal, y_true_abnormal], axis=0)

    print("推論を実行中...")
    # 再構成 - Use get_posterior_samples to handle both TF and PyTorch models
    from src.vae.vae_utils import get_posterior_samples
    X_recon = get_posterior_samples(vae, X_test)
    
    # 再構成誤差の計算 (各サンプルのMSE)
    # X_test shape: (N, T, D)
    # Error: mean((X - X_recon)^2) over axes (1, 2)
    reconstruction_errors = np.mean(np.square(X_test - X_recon), axis=(1, 2))
    
    errors = np.asarray(reconstruction_errors)
    print("min:", np.nanmin(errors), "max:", np.nanmax(errors))
    print("has inf:", np.isinf(errors).any(), "has nan:", np.isnan(errors).any())
    print("len:", len(errors))

    
    # --- 指標の計算 ---
    auc = roc_auc_score(y_true, reconstruction_errors)
    print(f"\nROC AUC スコア: {auc:.4f}")

    # しきい値の決定 (例: F1スコアが最大になる点を探索)
    thresholds = np.linspace(reconstruction_errors.min(), reconstruction_errors.max(), 100)
    best_f1 = 0
    best_threshold = 0
    
    for th in thresholds:
        y_pred = (reconstruction_errors > th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
            
    print(f"ベスト F1 スコア: {best_f1:.4f} (しきい値: {best_threshold:.6f})")
    
    y_pred_best = (reconstruction_errors > best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_best)
    print("混同行列:")
    print(cm)

     # ------------ 可視化用の処理（ヒスト & PR カーブ）------------
    if save_results_dir:
        # 有限値のみを可視化対象にする
        errors = np.asarray(reconstruction_errors)
        labels = np.asarray(y_true)
        finite_mask = np.isfinite(errors)
        vis_errors = errors[finite_mask]
        vis_labels = labels[finite_mask]

        # 上位 99.5% までにクリップして極端な外れ値を除外（ヒストグラム用）
        upper = np.percentile(vis_errors, 99.5)
        hist_mask = vis_errors <= upper
        vis_errors_hist = vis_errors[hist_mask]
        vis_labels_hist = vis_labels[hist_mask]

        normal_errors = vis_errors_hist[vis_labels_hist == 0]
        abnormal_errors = vis_errors_hist[vis_labels_hist == 1]

        # --- 再構成誤差のヒストグラム ---
        plt.figure(figsize=(10, 6))
        sns.histplot(
            normal_errors,
            label="Normal (正常)",
            kde=False,          # KDE は無効化して MemoryError を回避
            stat="density",
            bins=50,
            color="blue",
        )
        sns.histplot(
            abnormal_errors,
            label="Abnormal (異常)",
            kde=False,
            stat="density",
            bins=50,
            color="red",
        )
        plt.axvline(
            best_threshold,
            color="green",
            linestyle="--",
            label=f"Best Threshold ({best_threshold:.4f})",
        )
        plt.title("Reconstruction Error Distribution (再構成誤差の分布)")
        plt.xlabel("Mean Squared Error")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        hist_path = os.path.join(save_results_dir, "reconstruction_error_hist.png")
        plt.savefig(hist_path)
        plt.close()
        print(f"再構成誤差ヒストグラムを保存しました: {hist_path}")

        # --- Precision-Recall カーブ ---
        precisions, recalls, pr_thresholds = precision_recall_curve(
            y_true, reconstruction_errors
        )

        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker=".")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True)
        plt.tight_layout()

        pr_path = os.path.join(save_results_dir, "precision_recall_curve.png")
        plt.savefig(pr_path)
        plt.close()
        print(f"Precision-Recall カーブを保存しました: {pr_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimeVAEを用いた異常検知の評価")
    parser.add_argument("--model_dir", type=str, required=True, help="学習済みモデルとスケーラーが含まれるディレクトリ")
    parser.add_argument("--test_data_dir", type=str, required=True, help="test_normal.npz と test_abnormal.npz が含まれるディレクトリ")
    parser.add_argument("--vae_type", type=str, default="timeVAE", help="モデルの種類")
    parser.add_argument("--save_dir", type=str, default="./evaluation_results", help="評価結果を保存するディレクトリ")
    
    args = parser.parse_args()
    
    evaluate_anomaly_detection(
        model_dir=args.model_dir,
        test_data_dir=args.test_data_dir,
        vae_type=args.vae_type,
        save_results_dir=args.save_dir
    )
