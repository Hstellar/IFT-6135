import torch
from q2_sampler import svhn_sampler
from q2_model import Critic, Generator
from torch import optim
from torchvision.utils import save_image



def lp_reg(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. Also consider that the norm used is the L2 norm. This is important to consider,
    because we make the assumption that your implementation follows this notation when testing your function. ***

    :param x: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    :return: (FloatTensor) - shape: (1,) - Lipschitz penalty
    """
    #Reference - https://medium.com/mlearning-ai/how-to-improve-image-generation-using-wasserstein-gan-1297f449ca75
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    real_image = x
    fake_image = y
    batch_size, channel, height, width= real_image.shape
    # CT = ((x.to(device) - y.to(device))**2).to(device)
    # cons_reg = torch.max(torch.zeros(CT.size(), device=device), (CT)-1).mean()
    # return cons_reg.to(device)
    alpha= torch.rand(batch_size,1,1,1, device=device).repeat(1, channel, height, width)
    # interpolated image=randomly weighted average between a real and fake image
    #interpolated image ← alpha *real image  + (1 − alpha) fake image
    interpolatted_image=(alpha*real_image) + (1-alpha) * fake_image
    
    # calculate the critic score on the interpolated image
    # interpolatted_image.requires_grad = True
    interpolatted_image.retain_grad()
    interpolated_score= critic(interpolatted_image)
    
    # take the gradient of the score wrt to the interpolated image
    gradient= torch.autograd.grad(inputs=interpolatted_image,
                                  outputs=interpolated_score,
                                  retain_graph=True,
                                  create_graph=True,
                                  grad_outputs=torch.ones_like(interpolated_score), only_inputs=True)[0]
    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim=1)
    zeros = torch.zeros(gradient_norm.size(), device=device)
    gradient_penalty = (torch.max((gradient_norm -1), zeros)**2).mean()
    return gradient_penalty



def vf_wasserstein_distance(p, q, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """
    return torch.mean(critic(p))-torch.mean(critic(q))

def get_noise(n_samples, noise_dim, device='cuda'):
    '''
    Generate noise vectors from the random normal distribution with dimensions (n_samples, noise_dim),
    where
        n_samples: the number of samples to generate based on  batch_size
        noise_dim: the dimension of the noise vector
        device: device type can be cuda or cpu
    '''
    return  torch.randn(n_samples,noise_dim,1,1,device=device)


if __name__ == '__main__':
    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    data_root = './'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_iter = 50000 # N training iterations
    n_critic_updates = 5 # N critic updates per generator update
    lp_coeff = 10 # Lipschitz penalty coefficient
    train_batch_size = 64
    test_batch_size = 64
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 100
    cur_step = 0
    display_step = 500
    train_loader, valid_loader, test_loader = svhn_sampler(data_root, train_batch_size, test_batch_size)

    generator = Generator(z_dim=z_dim).to(device)
    critic = Critic().to(device)

    optim_critic = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

    # COMPLETE TRAINING PROCEDURE
    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)
    test_iter = iter(test_loader)
    for i in range(n_iter):
        generator.train()
        critic.train()
        for _ in range(n_critic_updates):
            try:
                data = next(train_iter)[0].to(device)
            except Exception:
                train_iter = iter(train_loader)
                data = next(train_iter)[0].to(device)
            cur_batch_size = data.shape[0]
            # print(cur_batch_size)
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = generator(fake_noise).to(device)
            # critic_fake_pred = critic(fake).reshape(-1).to(device)
            # critic_real_pred = critic(data).reshape(-1).to(device)
            # print(data.shape)
            # print(critic_fake_pred.shape)
            # print(critic_real_pred.shape)
            gp = lp_reg(data, fake, critic).to(device)
            # print(gp.is_cuda)
            critic_loss = -(vf_wasserstein_distance(data, fake, critic)) + lp_coeff *gp
            
            critic.zero_grad()
            #To make a backward pass and retain the intermediary results
            critic_loss.backward(retain_graph=True)
            # Update optimizer
            optim_critic.step()
             
            #####
            # train the critic model here
            #####

        #####
        # train the generator model here
        #####
        gen_fake= critic(fake).reshape(-1)
        gen_loss = -torch.mean(gen_fake)
        generator.zero_grad()
        gen_loss.backward()
        # Update optimizer
        optim_generator.step()

        # Save sample images
        if i % 100 == 0:
            print('gen_loss',gen_loss)
            print('critic loss', critic_loss)
            z = torch.randn(64, z_dim, device=device)
            imgs = generator(z)
            save_image(imgs, f'imgs_{i}.png', normalize=True, value_range=(-1, 1))


    # COMPLETE QUALITATIVE EVALUATION
     ## Visualization code ##
        # if cur_step % display_step == 0 and cur_step > 0:
        #     print(f"Step {cur_step}: Generator loss: {gen_loss}, critic loss: {critic_loss}")
        #     display_images(fake)
        #     display_images(data)
        #     gen_loss = 0
        #     critic_loss = 0
        # cur_step += 1
