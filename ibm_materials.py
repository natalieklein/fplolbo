"""
Trying a pretrained transformer model from https://huggingface.co/ibm/materials.selfies-ted, 
https://github.com/IBM/materials, or https://huggingface.co/ibm/materials.smi-ted
"""

# %%
from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration, BartTokenizer
import selfies as sf
import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

from functions import load_cwa_smiles, normalize_smiles, calculate_tanimoto_similarities

# %% Load CWA smiles
smiles = load_cwa_smiles(remove_chirality=False,remove_isomer=False)

# %% Encode smiles
selfies = []
for s in smiles:
    selfies.append(sf.encoder(s).replace("][", "] ["))

# %% SMILES -- this one seems to work well
# https://github.com/IBM/materials/blob/main/models/smi_ted/notebooks/smi_ted_encoder_decoder_example.ipynb
from inference.smi_ted_light.load import load_smi_ted

model_smi_ted = load_smi_ted(
    folder='inference/smi_ted_light',
    ckpt_filename='smi-ted-Light_40.pt'
)

smiles_norm = []
for s in smiles:
    smiles_norm.append(normalize_smiles(s,isomeric=False)) # runs with isomer but doesn't really work as well

# encode/decode
with torch.no_grad():
    encode_embeddings = model_smi_ted.encode(smiles_norm, return_torch=True)
    decoded_smiles = model_smi_ted.decode(encode_embeddings)

# %%
# Convert SMILES to RDKit molecule objects
mols1 = [Chem.MolFromSmiles(smiles) for smiles in smiles_norm]
mols2 = [Chem.MolFromSmiles(smiles) for smiles in decoded_smiles]

# Compute fingerprints for each molecule
fps1 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in mols1]
fps2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in mols2]

# Calculate Tanimoto similarities
tanimoto_similarities = calculate_tanimoto_similarities(fps1, fps2)

plt.figure()
plt.hist(tanimoto_similarities)
plt.show()

for s, s_d in zip(smiles_norm, decoded_smiles):
    print(s)
    print(s_d)





### SELFIES BELOW; not sure generation works?
# %% conditional gen version?
n_beams=5
in_data = selfies[:1]
model_name = "ibm/materials.selfies-ted"
model = BartForConditionalGeneration.from_pretrained(model_name)
#tokenizer = BartTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(in_data, return_tensors="pt", max_length=len(in_data[0]), truncation=True, padding='max_length')
output_ids = model.generate(inputs["input_ids"], max_length=len(in_data[0]), num_beams=n_beams, early_stopping=True)

# %%
for i in range(len(in_data)):
    output_text = tokenizer.decode(output_ids[i], skip_special_tokens=True)
    print(selfies[i])
    print(output_text)




# %% Trying example code from huggingface

# %% Load model/tokenizer (their example)
tokenizer = AutoTokenizer.from_pretrained("ibm/materials.selfies-ted")
model = AutoModel.from_pretrained("ibm/materials.selfies-ted")

# %% tokenize
token = tokenizer(selfies, return_tensors='pt', max_length=196, truncation=True, padding='max_length')
input_ids = token['input_ids']
attention_mask = token['attention_mask']

# %% embedding
encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
model_output = encoder_outputs.last_hidden_state

input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
model_output = sum_embeddings / sum_mask

# %% reconstruction
# Manually decode using the decoder
#decoder_input_ids = torch.tensor([[tokenizer.bos_token_id]])  # Start with BOS token
decoder_input_ids = input_ids[:, :1]  # Take the first token as the initial decoder input

output_ids = []

# Generate tokens iteratively
enc_out = encoder_outputs.last_hidden_state[[0]]
dec_in_id = decoder_input_ids[[0]]
for _ in range(200):  # Adjust loop limit for desired max length
    decoder_outputs = model.decoder(
        input_ids=dec_in_id, 
        encoder_hidden_states=enc_out
    )

    hidden_states = decoder_outputs.last_hidden_state[:, -1, :]  # Get the last token's hidden state
    logits = torch.matmul(hidden_states, model.shared.weight.T)  # Using the model's shared embedding layer

    
    # Get the last predicted token and append to outputs
    #next_token_id = decoder_outputs.logits[:, -1, :].argmax(dim=-1)
    next_token_id = logits.argmax(dim=-1)
    output_ids.append(next_token_id.item())
    decoder_input_ids = torch.cat([dec_in_id, next_token_id.unsqueeze(-1)], dim=-1)

    # Stop if EOS token is generated
    if next_token_id.item() == tokenizer.eos_token_id:
        break

# Decode output IDs to text
reconstructed_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print("Reconstructed Text:", reconstructed_text)

# %%
