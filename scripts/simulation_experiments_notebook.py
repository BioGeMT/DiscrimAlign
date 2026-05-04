#!/usr/bin/env python
# coding: utf-8

# In[87]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[1]:


from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import numpy as np
from numpy import random as rd
from matplotlib import pyplot as plt


# In[2]:


from src.estimalign import estimalign
from src.logit_link import logit_partial_scores
from src.optimization import create_powerstep, create_constant_step


# # Data

# In[4]:


from miRBench.dataset import list_datasets, get_dataset_df


# In[5]:


hejret_train = get_dataset_df(list_datasets()[0], split="train")
hejret_test = get_dataset_df(list_datasets()[0], split="test")


# In[6]:


mirlist = hejret_train['noncodingRNA']
mirlist = [Seq(seq) for seq in mirlist]
genelist = hejret_train['gene']
genelist = [Seq(seq).reverse_complement() for seq in genelist]


# In[7]:


hejret_train


# # Optimization

# ### Simple model on miRNA alignments:

# In[8]:


true_match = 1
true_mismatch = -1
true_gapopen = -1.2
true_gapext = -0.1


# In[9]:


aligner = PairwiseAligner()
aligner.mode = 'local'
aligner.open_gap_score = true_gapopen
aligner.extend_gap_score = true_gapext
aligner.match = true_match
aligner.mismatch = true_mismatch
# aligner.end_gap_score = 0


# In[10]:


mirlist[0]


# In[11]:


genelist[0]


# In[12]:


print(next(aligner.align(mirlist[1], genelist[1])))


# In[13]:


scores = np.array([aligner.score(a, b) for a, b in zip(mirlist, genelist)])


# In[14]:


plt.figure()
plt.hist(scores, bins=100)
plt.show()


# In[15]:


true_alpha = -9


# In[16]:


logit_scores = logit_partial_scores(scores, true_alpha)


# In[17]:


plt.figure()
plt.hist(logit_scores, bins=100)
plt.show()


# In[19]:


labels = rd.rand(len(mirlist))
labels = labels <= logit_scores
labels


# In[20]:


true_logL = np.sum(np.log(logit_scores[labels]))+np.sum(np.log(1-logit_scores[~labels]))
print('Sum of log-logit scores:', np.sum(np.log(logit_scores)))
print('True LogL:', true_logL)


# In[21]:


plt.figure()
plt.plot(scores, labels, 'r.', alpha=0.005)


# In[22]:


plt.figure()
plt.plot(logit_scores, labels, 'r.', alpha=0.005)


# In[36]:


const_step = create_constant_step(0.00001)
# powerstep = create_powerstep(0.00001, power=0.5, burnin=0)
# powerstep = create_powerstep(0.00001, power=-0.5, burnin=0)


# In[37]:


NITER = 50


# In[42]:


params = estimalign(mirlist, genelist, labels, 
                    stepfunction=const_step,
                    aligner_mode='local',
                    substitution_mode='simple',
                    gap_mode = 'affine',
                    verbose=True, max_iter=NITER,
                    stochastic_factor=0.001,
                    num_threads = 16)


# In[43]:


print(params['final_loglik'])


# In[44]:


print(params['final_loglik'])


# In[45]:


plt.figure()
plt.subplot(221)
plt.plot(np.arange(NITER), params['subgradient_l2_trajectory'])
plt.plot([0, NITER], [0, 0], 'k--')
plt.title('Subgradient L2 norm trajectory')


plt.subplot(222)
plt.plot(np.arange(NITER+1), params['loglik_trajectory'])
plt.plot([0, NITER+1], [true_logL, true_logL], 'k--')
plt.title('LogLikelihood trajectory')

plt.subplot(223)
plt.plot(np.arange(NITER//2, NITER), params['subgradient_l2_trajectory'][NITER//2:])
plt.plot([NITER//2, NITER], [0, 0], 'k--')
plt.title('Subgradient L2 norm trajectory')

plt.subplot(224)
plt.plot(np.arange(NITER//2, NITER+1), params['loglik_trajectory'][NITER//2:])
plt.plot([NITER//2, NITER+1], [true_logL, true_logL], 'k--')
plt.title('LogLikelihood trajectory')

plt.tight_layout()


# In[29]:


plt.figure()
plt.subplot(221)
plt.plot(np.arange(NITER), params['subgradient_l2_trajectory'])
plt.plot([0, NITER], [0, 0], 'k--')
plt.title('Subgradient L2 norm trajectory')


plt.subplot(222)
plt.plot(np.arange(NITER+1), params['loglik_trajectory'])
plt.plot([0, NITER+1], [true_logL, true_logL], 'k--')
plt.title('LogLikelihood trajectory')

plt.subplot(223)
plt.plot(np.arange(NITER//2, NITER), params['subgradient_l2_trajectory'][NITER//2:])
plt.plot([NITER//2, NITER], [0, 0], 'k--')
plt.title('Subgradient L2 norm trajectory')

plt.subplot(224)
plt.plot(np.arange(NITER//2, NITER+1), params['loglik_trajectory'][NITER//2:])
plt.plot([NITER//2, NITER+1], [true_logL, true_logL], 'k--')
plt.title('LogLikelihood trajectory')

plt.tight_layout()


# In[68]:


print(true_match, params['match_score'])
print(true_mismatch, params['mismatch_score'])
print(true_gapopen, params['open_gap_score'])
print(true_gapext, params['extend_gap_score'])
print(true_alpha, params['alpha'])


# ### Step function parameters experiment

# In[71]:


labels = rd.rand(len(mirlist)) <= logit_scores


# In[80]:


true_logL = np.sum(np.log(logit_scores[labels]))+np.sum(np.log(1-logit_scores[~labels]))


# In[72]:


steplengts = np.linspace(0.000005, 0.00005, num=10)
steplengts


# In[83]:


NITER = 10


# In[84]:


estimalign_results = []
for stepl in steplengts:
    const_step = create_constant_step(stepl)
    params = estimalign(mirlist, genelist, labels, 
                    stepfunction=const_step,
                    aligner_mode='local',
                    substitution_mode='simple',
                    gap_mode = 'affine',
                    verbose=False, max_iter=NITER,
                    stochastic_factor=0.001,
                    num_threads = 16)
    estimalign_results.append(params)


# In[91]:


plt.figure()
for params in estimalign_results:
    plt.plot(np.arange(NITER+1), params['loglik_trajectory'], alpha=0.5)
plt.plot([0, NITER], [true_logL, true_logL], 'k--')
plt.legend(steplengts)
plt.tight_layout()
# plt.savefig('path', dpi=160)


# In[90]:


plt.figure()
for params in estimalign_results:
    plt.plot(np.arange(NITER+1), params['loglik_trajectory'], alpha=0.5)
plt.plot([0, NITER], [true_logL, true_logL], 'k--')
plt.legend(steplengts)
plt.xlim(0, 5)
plt.tight_layout()


# ### Replicates

# In[64]:


REPS = 20
NITER = 5


# In[65]:


const_step = create_constant_step(0.00001)
# powerstep = create_powerstep(0.00001, power=0.5, burnin=0)
# powerstep = create_powerstep(0.00001, power=-0.5, burnin=0)


# In[66]:


estimalign_results = []
true_logLs = []
for _ in range(REPS):
    labels = rd.rand(len(mirlist)) <= logit_scores
    true_logL = np.sum(np.log(logit_scores[labels]))+np.sum(np.log(1-logit_scores[~labels]))
    true_logLs.append(true_logL)
    params = estimalign(mirlist, genelist, labels, 
                    stepfunction=const_step,
                    aligner_mode='local',
                    substitution_mode='simple',
                    gap_mode = 'affine',
                    verbose=False, max_iter=NITER,
                    stochastic_factor=0.001,
                    num_threads = 16)
    estimalign_results.append(params)


# In[85]:


plt.figure(figsize=(7.5, 2.1))
plt.subplot(121)
for params in estimalign_results:
    plt.plot(np.arange(NITER+1), params['loglik_trajectory'], alpha=0.2, color='b')
plt.subplot(122)
plt.plot([0, NITER], [0, 0], 'k--')
for params, tlL in zip(estimalign_results, true_logLs):
    plt.plot(np.arange(NITER+1), tlL - params['loglik_trajectory'], alpha=0.2, color='b')


# ### General matrix, affine gap penalty

# In[14]:


true_gapopen = -1.2
true_gapext = -0.1
true_substitution = substitution_matrices.Array(alphabet='ACTG', 
                                          data=np.array([
                                              [1, -0.3, -1, -0.8], 
                                              [-0.6, 1.2, -0.3, -1], 
                                              [-1.2, -0.4, 1, -0.8], 
                                              [-0.4, -1.4, -0.9, 1.3]]))


# In[15]:


aligner = PairwiseAligner()
aligner.mode = 'local'
aligner.open_gap_score = true_gapopen
aligner.extend_gap_score = true_gapext
aligner.substitution_matrix = true_substitution


# In[16]:


scores = np.array([aligner.score(a, b) for a, b in zip(mirlist, genelist)])


# In[17]:


plt.figure()
plt.hist(scores, bins=100)
plt.show()


# In[18]:


true_alpha = -12
logit_scores = logit_partial_scores(scores, true_alpha)


# In[19]:


plt.figure()
plt.hist(logit_scores, bins=100)
plt.show()


# In[20]:


labels = rd.rand(len(mirlist))
labels = labels <= logit_scores
true_logL = np.sum(np.log(logit_scores[labels]))+np.sum(np.log(1-logit_scores[~labels]))
print('Sum of log-logit scores:', np.sum(np.log(logit_scores)))
print('True LogL:', true_logL)


# In[21]:


const_step = create_constant_step(0.00005)
# powerstep = create_powerstep(0.00005, power=0.5, burnin=0)
powerstep = create_powerstep(0.00002, power=-0.1, burnin=0)


# In[22]:


NITER = 200


# In[23]:


params = estimalign(mirlist, genelist, labels, 
                    stepfunction=const_step,
                    aligner_mode='local',
                    substitution_mode='general',
                    gap_mode='affine', 
                    stochastic_factor=0.01,
                    verbose=True, max_iter=NITER,
                    num_threads = 24)


# In[24]:


print(params['final_loglik'])


# In[25]:


plt.figure()
plt.subplot(221)
plt.plot(np.arange(NITER), params['subgradient_l2_trajectory'])
plt.plot([0, NITER], [0, 0], 'k--')
plt.title('Subgradient L2 norm trajectory')


plt.subplot(222)
plt.plot(np.arange(NITER+1), params['loglik_trajectory'])
plt.plot([0, NITER+1], [true_logL, true_logL], 'k--')
plt.title('LogLikelihood trajectory')


plt.subplot(223)
plt.plot(np.arange(NITER//2, NITER), params['subgradient_l2_trajectory'][NITER//2:])
plt.plot([NITER//2, NITER], [0, 0], 'k--')
plt.title('Subgradient L2 norm trajectory')

plt.subplot(224)
plt.plot(np.arange(NITER//2, NITER+1), params['loglik_trajectory'][NITER//2:])
plt.plot([NITER//2, NITER+1], [true_logL, true_logL], 'k--')
plt.title('LogLikelihood trajectory')

plt.tight_layout()


# In[26]:


print(true_gapopen, params['open_gap_score'])
print(true_gapext, params['extend_gap_score'])
print(true_alpha, params['alpha'])
true_subs_vector = []
param_subs_vector = []
for char1 in true_substitution.alphabet:
    for char2 in true_substitution.alphabet:
        true_v = true_substitution[char1, char2]
        true_subs_vector.append(true_v)
        estim_v = params['substitution_matrix'][char1, char2]
        param_subs_vector.append(estim_v)
        print(char1, char2, true_v, estim_v)

print(np.corrcoef(true_subs_vector, param_subs_vector))
print(np.mean(np.abs(np.array(true_subs_vector)- np.array(param_subs_vector))))


# In[27]:


plt.figure()
plt.plot(true_subs_vector, param_subs_vector, 'r.')


# In[28]:


from src.optimization import get_initial_estimate


# In[29]:


get_initial_estimate(params['alignments'], labels, substitution_mode='general', gap_mode='affine',
                     alphabet=true_substitution.alphabet)


# In[30]:


true_substitution


# In[34]:


estim_substitution = true_substitution.copy()
for char1 in true_substitution.alphabet:
    for char2 in true_substitution.alphabet:
        estim_substitution[char1, char2] = params['substitution_matrix'][char1, char2]


# In[35]:


estim_substitution


# # Proteins

# ### Simulated alignments

# In[339]:


blosum62 = substitution_matrices.load('BLOSUM62')


# In[351]:


coupling = 2**blosum62/(len(blosum62.alphabet)**2)
coupling /= np.sum(coupling)


# In[355]:


aa_pairs = [(char1, char2) for char1 in blosum62.alphabet for char2 in blosum62.alphabet]
prob_vect = [coupling[aapair] for aapair in aa_pairs]


# In[363]:


for aa, prob in zip(aa_pairs, prob_vect):
    print(aa, prob)


# In[353]:


SETSIZE = 5000


# In[359]:


Alist = []
Blist = []
for _ in range(SETSIZE):
    pairs = rd.choice(len(aa_pairs), p=prob_vect, replace=True, size=250)
    pairs = [aa_pairs[i] for i in pairs]
    aseq = ''.join(x[0] for x in pairs)
    bseq = ''.join(x[1] for x in pairs)
    Alist.append(aseq)
    Blist.append(bseq)


# In[361]:


Blist


# ### Symmetric matrix on protein alignments:

# In[112]:


positive_id_tuples = []
close_negative_id_tuples = []
with open('./Proteomes/human_to_chicken_upto1000aa.blast') as h:
    for l in h:
        l = l.split('\t')
        evalue = float(l[-2])
        if evalue == 0:
            positive_id_tuples.append([l[0], l[1]])
        elif evalue > 0.0001:
            close_negative_id_tuples.append([l[0], l[1]])


# In[113]:


human_proteome = list(SeqIO.parse('Proteomes/GCF_000001405.40/up_to_1000.faa', 'fasta'))
chick_proteome = list(SeqIO.parse('Proteomes/GCF_016699485.2/up_to_1000.faa', 'fasta'))


# In[121]:


alphabet = 'ACDEFGHIKLMNPQRSTVWY'


# In[122]:


human_proteome = [seq for seq in human_proteome if set(str(seq.seq)).issubset(set(alphabet))]
chick_proteome = [seq for seq in chick_proteome if set(str(seq.seq)).issubset(set(alphabet))]


# In[124]:


human_prot_ids = [seq.id for seq in human_proteome]
chick_prot_ids = [seq.id for seq in chick_proteome]


# In[130]:


human_prot_ids_set = set(human_prot_ids)
chick_prot_ids_set = set(chick_prot_ids)


# In[131]:


positive_id_tuples = [t for t in positive_id_tuples if t[0] in human_prot_ids_set and t[1] in chick_prot_ids_set]
# close_negative_id_tuples = [t for t in close_negative_id_tuples if t[0] in human_prot_ids_set and t[1] in chick_prot_ids_set]


# In[286]:


SETSIZE = 5000


# In[287]:


# pos_set = rd.choice(len(positive_id_tuples), SETSIZE//2, replace=False)
# neg_set = rd.choice(len(close_negative_id_tuples), SETSIZE//2, replace=False)
# prot_dset = [positive_id_tuples[i] for i in pos_set] + [close_negative_id_tuples[i] for i in neg_set]
pos_set = rd.choice(len(positive_id_tuples), SETSIZE, replace=False)
prot_dset = [positive_id_tuples[i] for i in pos_set]


# In[288]:


human_list = [human_proteome[human_prot_ids.index(hpid)] for hpid, cpid in prot_dset]
chick_list = [chick_proteome[chick_prot_ids.index(cpid)] for hpid, cpid in prot_dset]


# In[289]:


blosum62 = substitution_matrices.load('BLOSUM62')


# In[290]:


blosum62 /= np.sqrt(np.sum(blosum62**2))


# In[293]:


aligner = PairwiseAligner()
aligner.mode = 'global'
aligner.substitution_matrix=blosum62
aligner.open_gap_score = -1
aligner.extend_gap_score = -0.1


# Example scores of alignments to visualize gap open and extend penalties for global alignment:

# In[294]:


test_aln = aligner.align('ATA', 'AA')
print(next(test_aln))
print(2*blosum62['A', 'A'] - 1, test_aln.score)
test_aln = aligner.align('ATTA', 'AA')
print(next(test_aln))
print(2*blosum62['A', 'A'] - 1 - 0.1, test_aln.score)


# In[295]:


scores = np.array([aligner.score(a, b) for a, b in zip(human_list, chick_list)])


# In[296]:


plt.figure()
plt.hist(scores, bins=100)
plt.show()


# In[297]:


print(next(aligner.align(human_list[0], chick_list[0])))


# In[299]:


logit_scores = logit_partial_scores(scores, -20)


# In[300]:


plt.figure()
plt.hist(logit_scores, bins=100)
plt.show()


# Expectation check:

# In[301]:


logL_distribution = []
for _ in range(5000):
    labels = rd.rand(len(human_list))
    labels = labels <= logit_scores
    true_logL = np.sum(np.log(logit_scores[labels]))+np.sum(np.log(1-logit_scores[~labels]))
    logL_distribution.append(true_logL)


# In[302]:


plt.figure()
plt.hist(logL_distribution, bins=100)
plt.show()


# In[303]:


EL = 0 
VL = 0
for ls in logit_scores:
    if 1e-24 < ls < 1-1e-24:
        EL += ls*np.log(ls) + (1-ls)*np.log(1-ls)
        VL += ls*(1-ls)*(np.log(ls)**2 + np.log(1-ls)**2) # Incorrect
SDL = np.sqrt(VL)


# In[304]:


print('Expected LogL:', EL)
print('Average LogL:', np.mean(logL_distribution))
print('STD LogL:', SDL)
print('Sample STD:', np.std(logL_distribution))


# Fitting:

# In[321]:


labels = rd.rand(len(human_list))
labels = labels <= logit_scores


# In[322]:


true_logL = np.sum(np.log(logit_scores[labels]))+np.sum(np.log(1-logit_scores[~labels]))


# In[323]:


print('Sum of log-logit scores:', np.sum(np.log(logit_scores)))
print('True LogL:', true_logL)


# In[324]:


plt.figure()
plt.plot(logit_scores, labels, 'r.', alpha=0.5)


# In[325]:


const_step = create_constant_step(0.00005)
powerstep = create_powerstep(0.000005, power=0.5, burnin=0) # step 0.00005 good for SETSIZE==1000
# powerstep = create_powerstep(0.0000001, power=-0.5, burnin=0)


# In[326]:


NITER = 50


# In[327]:


set('WEIMFASVCGKLTHDPRNQY') == set(alphabet) 


# In[328]:


params = estimalign(human_list, chick_list, labels, 
                    stepfunction=powerstep,
                    aligner_mode='global',
                    substitution_mode='general',
                    gap_mode='affine',
                    baseline_aligner=aligner,
                    stochastic_factor=0.0001,
                    verbose=True, max_iter=NITER,
                    num_threads = 16)


# In[329]:


print(params['final_loglik'])


# In[330]:


print(max(params['loglik_trajectory']))


# In[331]:


plt.figure()
plt.subplot(221)
plt.plot(np.arange(NITER), params['subgradient_l2_trajectory'])
plt.plot([0, NITER], [0, 0], 'k--')
plt.title('Subgradient L2 norm trajectory')


plt.subplot(222)
plt.plot(np.arange(NITER+1), params['loglik_trajectory'])
plt.plot([0, NITER+1], [true_logL, true_logL], 'k--')
plt.title('LogLikelihood trajectory')

plt.subplot(223)
plt.plot(np.arange(NITER//2, NITER), params['subgradient_l2_trajectory'][NITER//2:])
plt.plot([NITER//2, NITER], [0, 0], 'k--')
plt.title('Subgradient L2 norm trajectory')

plt.subplot(224)
plt.plot(np.arange(NITER//2, NITER+1), params['loglik_trajectory'][NITER//2:])
plt.plot([NITER//2, NITER+1], [true_logL, true_logL], 'k--')
plt.title('LogLikelihood trajectory')

plt.tight_layout()


# In[332]:


print(params['open_gap_score'], aligner.open_gap_score)
print(params['extend_gap_score'], aligner.extend_gap_score)


# In[333]:


params['alpha']


# In[334]:


blosum_vs_mine = []
for char1 in params['substitution_matrix'].alphabet:
    for char2 in params['substitution_matrix'].alphabet:
        if char1 != 'N' and char2 != 'N':
            blosum_vs_mine.append([char1, char2,blosum62[char1, char2], params['substitution_matrix'][char1, char2]])


# In[335]:


print('Blosum, Mine')
for i in rd.choice(len(blosum_vs_mine), 10, replace=False):
    print(blosum_vs_mine[i])


# In[336]:


np.corrcoef([x[2] for x in blosum_vs_mine], [x[3] for x in blosum_vs_mine])


# In[337]:


plt.figure()
plt.plot([x[2] for x in blosum_vs_mine], [x[3] for x in blosum_vs_mine], 'r.')


# In[338]:


plt.figure()
plt.plot(sorted([x[2] for x in blosum_vs_mine]), sorted([x[3] for x in blosum_vs_mine]), 'r.')


# In[ ]:




