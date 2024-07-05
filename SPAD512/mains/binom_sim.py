import numpy as np
import matplotlib.pyplot as plt
from SPAD512.sr.binom import PoissonBinomial2
import time

def post_var_lambda(lambdas,Ns,N=3,nbatch=100,num_samples=100,nexpose=1000,zeta=0.01):
    posts = []
    for lambd in lambdas:
        batch_posts = []
        for n in range(nbatch):
            #print(f'Batch {n}')
            binomial_data = np.random.binomial(N, zeta, size=nexpose)
            poisson_data = np.random.poisson(lambd, size=nexpose)
            observed_data = binomial_data + poisson_data
            model = PoissonBinomial2(observed_data,lambd=lambd,
                                     zeta_mean=0.01,zeta_std=0.005)
            post = model.integrate(num_samples,Ns)
            post = np.array(post)
            post = post/np.sum(post)
            batch_posts.append(post)
        batch_posts = np.array(batch_posts)
        avg_post = np.squeeze(np.mean(batch_posts,axis=0))
        posts.append(avg_post)
    posts = np.array(posts)
    return posts

Ns = np.arange(1,30,1)
lambdas = [0.0,0.0075,0.01]
start = time.time()
posts0 = post_var_lambda(lambdas,Ns,N=7,num_samples=100)
print(time.time() - start)
posts1 = post_var_lambda(lambdas,Ns,N=10,num_samples=100)
posts2 = post_var_lambda(lambdas,Ns,N=12,num_samples=100)

fig,ax=plt.subplots(1,3,figsize=(10,4),sharex=True,sharey=True)
ax[0].bar(Ns, posts0[0], alpha=0.3, color='red', label=r'$\lambda$='+f'{lambdas[0]} cts/pulse')
ax[0].bar(Ns, posts0[1], alpha=0.3, color='blue', label=r'$\lambda$='+f'{lambdas[1]} cts/pulse')
ax[0].bar(Ns, posts0[2], alpha=0.3, color='lime', label=r'$\lambda$='+f'{lambdas[2]} cts/pulse')
ax[0].set_title('True N=7')
ax[1].bar(Ns, posts1[0], alpha=0.3, color='red', label=r'$\lambda$='+f'{lambdas[0]}')
ax[1].bar(Ns, posts1[1], alpha=0.3, color='blue', label=r'$\lambda$='+f'{lambdas[1]}')
ax[1].bar(Ns, posts1[2], alpha=0.3, color='lime', label=r'$\lambda$='+f'{lambdas[2]}')
ax[1].set_title('True N=10')
ax[2].bar(Ns, posts2[0], alpha=0.3, color='red', label=r'$\lambda$='+f'{lambdas[0]}')
ax[2].bar(Ns, posts2[1], alpha=0.3, color='blue', label=r'$\lambda$='+f'{lambdas[1]}')
ax[2].bar(Ns, posts2[2], alpha=0.3, color='lime', label=r'$\lambda$='+f'{lambdas[2]}')
ax[2].set_title('True N=12')
ax[0].set_ylabel('Posterior Probability')
ax[0].legend()
for axi in ax.ravel():
    axi.set_xlim([0,30])
    axi.set_xticks(np.arange(0,30,2))
    axi.set_xlabel('N')
plt.tight_layout()
plt.show()

