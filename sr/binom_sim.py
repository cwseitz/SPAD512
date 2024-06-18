import numpy as np
import matplotlib.pyplot as plt
from binom import PoissonBinomial

def post_var_lambda(lambdas,Ns,N=3,num_samples=1000,zeta=0.05):
    posts = []
    for lambd in lambdas:
        print(f'Background rate: {lambd}')
        binomial_data = np.random.binomial(N, zeta, size=1000)
        poisson_data = np.random.poisson(lambd, size=1000)
        observed_data = binomial_data + poisson_data
        model = PoissonBinomial(observed_data,lambd=lambd,zeta_std=0.01)
        post = [model.integrate(num_samples,n,approx=True) for n in Ns]
        post = np.array(post)
        post = post/np.sum(post)
        mapp = np.argmax(post)
        H = -np.sum(post*np.log(1e-8+post))
        posts.append(post)
    posts = np.array(posts)
    return posts

Ns = np.arange(1,20,1)
lambdas = [0.0,0.02,5.0]
posts0 = post_var_lambda(lambdas,Ns,N=1)
posts1 = post_var_lambda(lambdas,Ns,N=2)
posts2 = post_var_lambda(lambdas,Ns,N=5)

fig,ax=plt.subplots(1,3,figsize=(10,4),sharex=True,sharey=True)
ax[0].bar(Ns, posts0[0], alpha=0.3, color='red', label=r'$\lambda$='+f'{lambdas[0]} cts')
ax[0].bar(Ns, posts0[1], alpha=0.3, color='blue', label=r'$\lambda$='+f'{lambdas[1]} cts')
ax[0].bar(Ns, posts0[2], alpha=0.3, color='lime', label=r'$\lambda$='+f'{lambdas[2]} cts')
ax[0].set_title('True N=1')
ax[1].bar(Ns, posts1[0], alpha=0.3, color='red', label=r'$\lambda$='+f'{lambdas[0]}')
ax[1].bar(Ns, posts1[1], alpha=0.3, color='blue', label=r'$\lambda$='+f'{lambdas[1]}')
ax[1].bar(Ns, posts1[2], alpha=0.3, color='lime', label=r'$\lambda$='+f'{lambdas[2]}')
ax[1].set_title('True N=2')
ax[2].bar(Ns, posts2[0], alpha=0.3, color='red', label=r'$\lambda$='+f'{lambdas[0]}')
ax[2].bar(Ns, posts2[1], alpha=0.3, color='blue', label=r'$\lambda$='+f'{lambdas[1]}')
ax[2].bar(Ns, posts2[2], alpha=0.3, color='lime', label=r'$\lambda$='+f'{lambdas[2]}')
ax[2].set_title('True N=5')
ax[0].set_ylabel('Posterior Probability')
ax[0].legend()
for axi in ax.ravel():
    axi.set_xlim([0,15])
    axi.set_xticks(np.arange(0,15,2))
    axi.set_xlabel('N')
plt.tight_layout()
plt.show()

