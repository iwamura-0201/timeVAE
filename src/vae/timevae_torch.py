import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# from vae.vae_base import BaseVariationalAutoencoder, Sampling  # Removing TF dependencies

class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

class TrendLayer(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, trend_poly):
        super(TrendLayer, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.trend_dense1 = nn.Linear(self.latent_dim, self.feat_dim * self.trend_poly)
        self.trend_dense2 = nn.Linear(self.feat_dim * self.trend_poly, self.feat_dim * self.trend_poly)

    def forward(self, z):
        trend_params = F.relu(self.trend_dense1(z))
        trend_params = self.trend_dense2(trend_params)
        trend_params = trend_params.view(-1, self.feat_dim, self.trend_poly)

        lin_space = torch.arange(0, float(self.seq_len), 1, device=z.device) / self.seq_len 
        poly_space = torch.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], dim=0) 

        trend_vals = torch.matmul(trend_params, poly_space) 
        trend_vals = trend_vals.permute(0, 2, 1) 
        return trend_vals


class SeasonalLayer(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, custom_seas):
        super(SeasonalLayer, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.custom_seas = custom_seas

        self.dense_layers = nn.ModuleList([
            nn.Linear(latent_dim, feat_dim * num_seasons)
            for num_seasons, len_per_season in custom_seas
        ])
        

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        season_indexes = torch.arange(num_seasons).unsqueeze(1) + torch.zeros(
            (num_seasons, len_per_season), dtype=torch.int32
        )
        season_indexes = season_indexes.view(-1)
        season_indexes = season_indexes.repeat(self.seq_len // len_per_season + 1)[: self.seq_len]
        return season_indexes

    def forward(self, z):
        N = z.shape[0]
        ones_tensor = torch.ones((N, self.feat_dim, self.seq_len), dtype=torch.int32, device=z.device)

        all_seas_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)
            season_params = season_params.view(-1, self.feat_dim, num_seasons)

            season_indexes_over_time = self._get_season_indexes_over_seq(
                num_seasons, len_per_season
            ).to(z.device)

            dim2_idxes = ones_tensor * season_indexes_over_time.view(1, 1, -1)
            season_vals = torch.gather(season_params, 2, dim2_idxes.long()) # Ensure indices are long

            all_seas_vals.append(season_vals)

        all_seas_vals = torch.stack(all_seas_vals, dim=-1) 
        all_seas_vals = torch.sum(all_seas_vals, dim=-1)  
        all_seas_vals = all_seas_vals.permute(0, 2, 1)  

        return all_seas_vals
    

class LevelModel(nn.Module):
    def __init__(self, latent_dim, feat_dim, seq_len):
        super(LevelModel, self).__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.level_dense1 = nn.Linear(self.latent_dim, self.feat_dim)
        self.level_dense2 = nn.Linear(self.feat_dim, self.feat_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        level_params = self.relu(self.level_dense1(z))
        level_params = self.level_dense2(level_params)
        level_params = level_params.view(-1, 1, self.feat_dim)

        ones_tensor = torch.ones((1, self.seq_len, 1), dtype=torch.float32, device=z.device)
        level_vals = level_params * ones_tensor
        return level_vals


class ResidualConnection(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim, encoder_last_dense_dim):
        super(ResidualConnection, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        
        self.dense = nn.Linear(latent_dim, encoder_last_dense_dim)
        self.deconv_layers = nn.ModuleList()
        in_channels = hidden_layer_sizes[-1]
        
        for i, num_filters in enumerate(reversed(hidden_layer_sizes[:-1])):
            self.deconv_layers.append(
                nn.ConvTranspose1d(in_channels, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            in_channels = num_filters
            
        self.deconv_layers.append(
            nn.ConvTranspose1d(in_channels, feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        # Check Output padding logic if needed, simplistically implemented assumes input length was perfectly divisible or matched
        self.final_dense = nn.Linear(feat_dim * seq_len, seq_len * feat_dim) # Fallback to dense if shape mismatch, or just dense correction

    def forward(self, z):
        batch_size = z.size(0)
        x = F.relu(self.dense(z))
        
        # Reshape to (Batch, Channels, Length) for Conv1d
        # Based on encoder: (Batch, Hidden[-1], Length_Encoded)
        # We need to reshape x to match expected input for deconv
        # Here we simplify: assume global pooling was not used, but Flatten was.
        # So we need to recover the length.
        
        # NOTE: This implementation might need adjustment based on exact encoder output shape.
        # Assuming encoder_last_dense_dim = hidden_layer_sizes[-1] * compressed_len
        
        compressed_len = x.shape[1] // self.hidden_layer_sizes[-1]
        x = x.view(batch_size, self.hidden_layer_sizes[-1], -1) 
        
        for deconv in self.deconv_layers[:-1]:
            x = F.relu(deconv(x))
        x = F.relu(self.deconv_layers[-1](x))
        
        # At this point x is (Batch, FeatDim, Length)
        # If length doesn't match seq_len due to padding/strides, we interpolate or use dense
        if x.shape[2] != self.seq_len:
             x = F.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
        
        x = x.permute(0, 2, 1) # (Batch, Length, FeatDim)
        residuals = x 
        return residuals
    

class TimeVAEEncoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim):
        super(TimeVAEEncoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = nn.ModuleList()
        
        in_channels = feat_dim
        for num_filters in hidden_layer_sizes:
            self.layers.append(nn.Conv1d(in_channels, num_filters, kernel_size=3, stride=2, padding=1))
            self.layers.append(nn.ReLU())
            in_channels = num_filters

        self.encoder_last_dense_dim = self._get_last_dense_dim(seq_len, feat_dim)
        
        self.z_mean = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.z_log_var = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.sampling = Sampling()

    def forward(self, x):
        # x: (Batch, Length, Feat) -> (Batch, Feat, Length)
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        
        x = x.flatten(1)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z
    
    def _get_last_dense_dim(self, seq_len, feat_dim):
        with torch.no_grad():
            x = torch.randn(1, feat_dim, seq_len)
            for layer in self.layers:
                x = layer(x)
            return x.numel()

class TimeVAEDecoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim, trend_poly=0, custom_seas=None, use_residual_conn=True, encoder_last_dense_dim=None):
        super(TimeVAEDecoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn
        self.encoder_last_dense_dim = encoder_last_dense_dim
        self.level_model = LevelModel(self.latent_dim, self.feat_dim, self.seq_len)

        if use_residual_conn:
            self.residual_conn = ResidualConnection(seq_len, feat_dim, hidden_layer_sizes, latent_dim, encoder_last_dense_dim)

    def forward(self, z):
        outputs = self.level_model(z)
        if self.trend_poly is not None and self.trend_poly > 0:
            trend_vals = TrendLayer(self.seq_len, self.feat_dim, self.latent_dim, self.trend_poly).to(z.device)(z)
            outputs += trend_vals

        # custom seasons
        if self.custom_seas is not None and len(self.custom_seas) > 0:
            cust_seas_vals = SeasonalLayer(self.seq_len, self.feat_dim, self.latent_dim, self.custom_seas).to(z.device)(z)
            outputs += cust_seas_vals

        if self.use_residual_conn:
            residuals = self.residual_conn(z)
            outputs += residuals

        return outputs


class TimeVAE(nn.Module):
    model_name = "TimeVAE"

    def __init__(
        self,
        seq_len,
        feat_dim,
        latent_dim,
        hidden_layer_sizes=None,
        trend_poly=0,
        custom_seas=None,
        use_residual_conn=True,
        reconstruction_wt=3.0,
        **kwargs,
    ):
        super(TimeVAE, self).__init__()

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50, 100, 200]

        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn
        self.reconstruction_wt = reconstruction_wt

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

    def _get_encoder(self):
        return TimeVAEEncoder(self.seq_len, self.feat_dim, self.hidden_layer_sizes, self.latent_dim)

    def _get_decoder(self):
        return TimeVAEDecoder(self.seq_len, self.feat_dim, self.hidden_layer_sizes, self.latent_dim, self.trend_poly, self.custom_seas, self.use_residual_conn, self.encoder.encoder_last_dense_dim)

    def get_prior_samples(self, num_samples, device='cpu'):
        self.to(device)
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)
        return samples.cpu().numpy()

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var

    def loss_function(self, x, x_recon, z_mean, z_log_var):
        # Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') # Sum over batch and pixels
        
        # KL Divergence
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        return self.reconstruction_wt * recon_loss + kl_loss, recon_loss, kl_loss

    def fit(self, train_data, epochs=100, batch_size=32, learning_rate=1e-3, device='cpu', verbose=1):
        """
        Train the VAE model.
        train_data: numpy array (N, T, D)
        """
        self.to(device)
        self.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        dataset = TensorDataset(torch.from_numpy(train_data).float())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            
            for (x_batch,) in dataloader:
                x_batch = x_batch.to(device)
                
                optimizer.zero_grad()
                x_recon, z_mean, z_log_var = self(x_batch)
                
                loss, recon_loss, kl_loss = self.loss_function(x_batch, x_recon, z_mean, z_log_var)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                
            avg_loss = total_loss / len(dataset)
            
            if verbose and (epoch + 1) % verbose == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Recon Loss: {total_recon_loss/len(dataset):.4f}, KL Loss: {total_kl_loss/len(dataset):.4f}")

    def predict(self, data, batch_size=32, device='cpu'):
        self.to(device)
        self.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).float())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        recon_list = []
        with torch.no_grad():
            for (x_batch,) in dataloader:
                x_batch = x_batch.to(device)
                x_recon, _, _ = self(x_batch)
                recon_list.append(x_recon.cpu().numpy())
                
        return np.concatenate(recon_list, axis=0)

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, f"{self.model_name}_weights.pth"))

        if self.custom_seas is not None:
             # Convert to list if it's not (though in Torch we kept it as list of tuples usually)
             pass

        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "reconstruction_wt": self.reconstruction_wt,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "trend_poly": self.trend_poly,
            "custom_seas": self.custom_seas,
            "use_residual_conn": self.use_residual_conn,
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)

    @classmethod
    def load(cls, model_dir: str, device='cpu') -> "TimeVAE":
        params_file = os.path.join(model_dir, f"{cls.model_name}_parameters.pkl")
        dict_params = joblib.load(params_file)
        vae_model = TimeVAE(**dict_params)
        weights_path = os.path.join(model_dir, f"{cls.model_name}_weights.pth")
        vae_model.load_state_dict(torch.load(weights_path, map_location=device))
        vae_model.to(device)
        return vae_model
