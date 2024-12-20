"""
Load SELFIES VAE from repo and check it out.

Author: Natalie Klein

Guacamol data from: https://figshare.com/projects/GuacaMol/56639

Questions:
- We should be able to sample in latent space, and sample in output space... right?
    - Latent space: yes, it returns a sample in forward
    - Output space: less clear whether it's deterministic.
- Why are similarities not so great even on the (supposed) training set?
    - not sure exactly which subset of guacamol they used
- Is it ok to ignore stereochemistry and stuff? (Not in valid tokens)
    - In theory SELFIES can handle it... maybe we should retrain the VAE with an expanded datset such as PubChem?
- There is a guacamol set in the lolbo repo
    
"""
# %%
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import selfies as sf
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from lolbo.molecule_objective import MoleculeObjective

from functions import canon_smiles, load_cwa_smiles

np.random.seed(42)

state_path = "lolbo/utils/mol_utils/selfies_vae/state_dict/SELFIES-VAE-state-dict.pt"

#%%
# has method vae_forward that takes smiles strings
# has method vae_decode that returns smiles strings
mo = MoleculeObjective(path_to_vae_statedict=state_path)
mo.initialize_vae()

#mo.dataobj.vocab

# %% load guacamol data
with open('data/guacamol_v1_all.smiles','r') as f:
    guac_smiles = f.readlines()
guac_smiles = [g.strip() for g in guac_smiles]

# Remove stuff not in the vocab -- is there a better way? what is the data set they used?
guac_clean = []
for smile in list(np.random.choice(guac_smiles, 100)):
    try:
        selfie = sf.encoder(smile)
        tokens = mo.dataobj.tokenize_selfies([selfie])[0]
        for t in tokens:
            if t not in mo.dataobj.vocab:
                raise KeyError
        guac_clean.append(smile)
    except:
        pass

# %% apply  model to guacamol data; TODO use batching
guac_r, _ = mo.vae_forward(guac_clean)
guac_pred = mo.vae_decode(guac_r)
#guac_pred = []
#for gc in guac_clean:
#    r, _ = mo.vae_forward([gc])
#    guac_pred.append(mo.vae_decode(r)[0])

smiles = load_cwa_smiles()

# %% exclude ones that don't match the vocab
smiles_clean = []
for smile in smiles:
    try:
        selfie = sf.encoder(smile)
        tokens = mo.dataobj.tokenize_selfies([selfie])[0]
        for t in tokens:
            if t not in mo.dataobj.vocab:
                print(t)
                raise KeyError
        smiles_clean.append(smile)
    except KeyError as e:
        print(e)
        print(smile)
        print(selfie)
        pass

# %% apply model
smiles_r, _ = mo.vae_forward(smiles_clean)
smiles_pred = mo.vae_decode(smiles_r)

# %% Try to pass it in; doing one at a time to catch errors
# smiles_input = []
# smiles_pred = []
# for s in smiles:
#     try:
#         r, _ = mo.vae_forward([s])
#         spred = mo.vae_decode(r)
#         smiles_pred.append(spred[0])
#         smiles_input.append(s)
#     except KeyError:
#         continue

# %% Tanimoto similarity
guac_tani_sim = []
for s, spred in zip(guac_clean, guac_pred):
    # Convert to RDKit molecules
    mol1 = Chem.MolFromSmiles(s)
    mol2 = Chem.MolFromSmiles(spred)

    # Generate Morgan fingerprints (circular fingerprints)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=3, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=3, nBits=2048)

    # Calculate Tanimoto similarity
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

    guac_tani_sim.append(similarity)

tani_sim = []
for s, spred in zip(smiles_clean, smiles_pred):
    # Convert to RDKit molecules
    mol1 = Chem.MolFromSmiles(s)
    mol2 = Chem.MolFromSmiles(spred)

    # Generate Morgan fingerprints (circular fingerprints)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=3, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=3, nBits=2048)

    # Calculate Tanimoto similarity
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

    tani_sim.append(similarity)

plt.figure()
plt.hist(tani_sim,bins=20,alpha=0.7,label='CWA')
plt.hist(guac_tani_sim,bins=20,alpha=0.7,label='Guacamol')
plt.title('Tanimoto similarity')
plt.axvline(0.85)
plt.legend()
plt.xlim([0,1])
plt.show()

# %% TSNE representation in latent space
guac_r = guac_r.detach().cpu().numpy()
smiles_r = smiles_r.detach().cpu().numpy()
r_all  = np.concatenate([guac_r, smiles_r], 0)
# Create a TSNE object
tsne = TSNE(n_components=2, random_state=0)
# Fit and transform the data
embed = tsne.fit_transform(r_all)
embed_guac = embed[:len(guac_r)]
embed_smiles = embed[len(guac_r):]

plt.figure()
plt.plot(embed_guac[:, 0], embed_guac[:, 1], 'k.', label='Guacamol')
plt.plot(embed_smiles[:, 0], embed_smiles[:, 1], 'r.', alpha=0.7, label='CWA')
plt.legend()
plt.title('tSNE')
plt.show()

# %% PCA representation in latent space
pca = PCA(n_components=2).fit(r_all)
embed_guac = pca.transform(guac_r)
embed_smiles = pca.transform(smiles_r)

plt.figure()
plt.plot(embed_guac[:, 0], embed_guac[:, 1], 'k.', label='Guacamol')
plt.plot(embed_smiles[:, 0], embed_smiles[:, 1], 'r.', alpha=0.7, label='CWA')
plt.legend()
plt.title('PCA')
plt.show()

# %% Frechet ChemNet distance between guac_smiles samples, smiles_clean (samples?)
from fcd import get_fcd, load_ref_model, canonical_smiles, get_predictions, calculate_frechet_distance
# Load chemnet model
model = load_ref_model()
cross_fcd_scores = []
for _ in range(50):
    # get canonical smiles and filter invalid ones
    sample1 = np.random.choice(guac_smiles, 100, replace=False)
    can_sample1 = [w for w in canonical_smiles(sample1) if w is not None]
    sample2 = np.random.choice(smiles_clean, 100, replace=True)
    can_sample2 = [w for w in canonical_smiles(sample2) if w is not None]

    fcd_score = get_fcd(can_sample1, can_sample2, model)
    cross_fcd_scores.append(fcd_score)

# %%
within_fcd_scores = []
for _ in range(50):
    # get canonical smiles and filter invalid ones
    sample1 = np.random.choice(guac_smiles, 100, replace=False)
    can_sample1 = [w for w in canonical_smiles(sample1) if w is not None]
    sample2 = np.random.choice(guac_smiles, 100, replace=True)
    can_sample2 = [w for w in canonical_smiles(sample2) if w is not None]

    fcd_score = get_fcd(can_sample1, can_sample2, model)
    within_fcd_scores.append(fcd_score)
# %%

plt.figure()
plt.hist(cross_fcd_scores, label='CWA/GuacaMol', alpha=0.7)
plt.hist(within_fcd_scores, label='GuacaMol/GuacaMol', alpha=0.7)
plt.legend()
plt.xlabel('Frechet ChemNet Distance')
plt.show()

# %%
