import clip
import dnnlib
import legacy
import torch 
import torchvision
import math
from torchvision.utils import make_grid
import PIL 
import matplotlib.pyplot as plt

def load_models():
    # Load clip model
    clip_model, clip_preprocess = clip.load("ViT-B/32")
    clip_model.eval()
    clip_preprocess = torchvision.transforms.Compose([clip_preprocess.transforms[0],
                                                      clip_preprocess.transforms[1],
                                                      clip_preprocess.transforms[4],])
    # Load style-gan2 model trained on FFHQ dataset
    network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
    with dnnlib.util.open_url(network_pkl) as f:
        stylegan2_model = legacy.load_network_pkl(f)['G_ema'] # type: ignore
        stylegan2_model.eval()
    return stylegan2_model, clip_model, clip_preprocess

class GaussianDistribution(torch.nn.Module):
  def __init__(self, z_dim):
    super(GaussianDistribution, self).__init__()
    self.mean_latent = torch.nn.Parameter(torch.zeros(stylegan2_model.z_dim),requires_grad=True)
    # initialize log standard deviation to small value for faster convergence rate udring optimization ( sacrificing diversity)
    self.log_std_latent = torch.nn.Parameter(torch.ones(stylegan2_model.z_dim)*(-2.0),requires_grad=True)

  def log_prob(self, z):
    n_samples = z.shape[0]
    mean = self.mean_latent.unsqueeze(0).repeat(n_samples,1)
    log_std = self.log_std_latent.unsqueeze(0).repeat(n_samples,1)
    return -0.5 * ((z - mean) / torch.exp(log_std)) ** 2 - log_std - math.log(math.sqrt(2 * math.pi))

  def sample(self, n_samples):
    log_std = self.log_std_latent.unsqueeze(0).repeat(n_samples,1)
    mean = self.mean_latent.unsqueeze(0).repeat(n_samples,1)
    return mean + torch.exp(log_std) * torch.randn_like(mean)
  
def description2image_optimization(textual_description, 
                                  latent_distribution,
                                      clip_model, 
                                      clip_preprocess, 
                                      stylegan_model,
                                      num_optimization_steps=1000,
                                      img_batch_size=12,
                                      learning_rate=1e-2,
                                      use_cuda = True):
    if(use_cuda):
      stylegan_model.cuda()
      clip_model.cuda()
                          
    # Get textual description embedding
    with torch.no_grad():
      text_tokens = clip.tokenize([textual_description])
      if(use_cuda):
        text_tokens = text_tokens.cuda()
      textual_description_embedding = clip_model.encode_text(text_tokens).float()

    # Initialize latent learned latent distribution
    if(use_cuda):
      latent_distribution = latent_distribution.cuda()
    
    # Optimization
    optimizer = torch.optim.Adam(latent_distribution.parameters(), lr=learning_rate)
    for step in range(0,num_optimization_steps):
      optimizer.zero_grad()                                                                          
      # Generate images with style gan from learned distribution
      with torch.no_grad():
        z = latent_distribution.sample(img_batch_size) #TODO: check this
        label_embedding = torch.zeros((img_batch_size, stylegan_model.c_dim))
        if(use_cuda):
          z = z.cuda()
          label_embedding = label_embedding.cuda()
        gen_imgs = stylegan_model(z, label_embedding, truncation_psi=1, noise_mode='random')
        # Preprocess images for clip
        clip_imgs = (gen_imgs*0.5+1.0).clamp(0.0,1.0)
        clip_imgs = clip_preprocess(clip_imgs)
        
        # Get  embeddings from generated images
        image_embeddings = clip_model.encode_image(clip_imgs)

        # Compute embedding similarity
        embeddings_similarity = torch.cosine_similarity(textual_description_embedding,
                                                       image_embeddings)

      # Optimize latent vector distribution to  
      latent_log_prob = latent_distribution.log_prob(z).mean(dim=1)
      loss = -latent_log_prob*embeddings_similarity
      loss = loss.mean()
      loss.backward()
      optimizer.step()
      if(step%10==0):
        print("Step: {}, Loss: {}, Embeddings Similiarity: {}".format(step, 
                                                        loss.item(), 
                                                        embeddings_similarity.mean().item()))
        
import warnings
warnings.filterwarnings('ignore')

use_cuda= True    
description = 'Blonde male wearing sunglasses and lipstick.'
stylegan2_model, clip_model, clip_preprocess = load_models()
latent_distribution = GaussianDistribution(stylegan2_model.z_dim)
description2image_optimization(description, latent_distribution, clip_model, clip_preprocess, stylegan2_model, use_cuda=use_cuda)




