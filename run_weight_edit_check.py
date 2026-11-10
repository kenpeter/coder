import torch
import json
from datetime import datetime

# Load the SAN model checkpoint
st = torch.load('/home/kenpeter/work/coder/checkpoints/san/san_latest.pt', map_location='cpu', weights_only=False)
sd = st['model_state_dict']

# REAL weight editing: subtract 0.01 from weight tensor (phi_pre)
first_key = list(sd.keys())[0]
original_weight = sd[first_key].clone()

# Apply the edit
edit_amount = 0.01
sd[first_key] = original_weight - edit_amount * torch.sign(original_weight)

# Create edited checkpoint
st_edited = st.copy()
st_edited['model_state_dict'] = sd

# Compute metrics
weight_change_magnitude = (sd[first_key] - original_weight).abs().mean().item()
max_weight_change = (sd[first_key] - original_weight).abs().max().item()

# Save the edited checkpoint with a new name
import os
os.makedirs('/home/kenpeter/work/coder/checkpoints/san/', exist_ok=True)
torch.save(st_edited, '/home/kenpeter/work/coder/checkpoints/san/san_edited.pt')

# Save results
results = {
    'original_loss': float(st['loss']),
    'edited_loss_note': 'metadata stays 1.0756 (expected behavior)',
    'weight_key': first_key,
    'edit_amount': edit_amount,
    'weight_change_magnitude': weight_change_magnitude,
    'max_weight_change': max_weight_change,
    'original_weight_mean': float(original_weight.mean().item()),
    'edited_weight_mean': float(sd[first_key].mean().item()),
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'edit_type': 'final_check_test'
}

with open('/home/kenpeter/work/coder/weight_edit_final_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Weight editing cycle completed')
print(f'Original checkpoint loss: {st["loss"]}')
print(f'Edited checkpoint saved: san_edited.pt')
print(f'Weight change magnitude: {weight_change_magnitude:.8f}')
print(f'Weight means: original={float(original_weight.mean().item()):.10f}, edited={float(sd[first_key].mean().item()):.10f}')
