import torch
import json
from datetime import datetime

# Load the SAN model checkpoint (preserve original)
st = torch.load('/home/kenpeter/work/coder/checkpoints/san/san_latest.pt', map_location='cpu', weights_only=False)
sd = st['model_state_dict']

# Track what was edited
edit_summary = {
    'total_groups': len(sd),
    'edited_groups': 0,
    'weight_changes': {},
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'edit_amount': 0.01,
    'edit_type': 'edit_all_weights'
}

# REAL weight editing: subtract 0.01 from ALL weight tensors
for key in sd.keys():
    original_weight = sd[key].clone()
    # Apply edit: subtract 0.01 * sign of original weight
    sd[key] = original_weight - 0.01 * torch.sign(original_weight)
    
    # Compute change metrics
    change_magnitude = (sd[key] - original_weight).abs().mean().item()
    max_change = (sd[key] - original_weight).abs().max().item()
    
    # Store summary
    edit_summary['edited_groups'] += 1
    edit_summary['weight_changes'][key] = {
        'change_magnitude': change_magnitude,
        'max_change': max_change,
        'original_mean': float(original_weight.mean().item()),
        'edited_mean': float(sd[key].mean().item())
    }

# Create edited checkpoint
st_edited = st.copy()
st_edited['model_state_dict'] = sd

# Save edited checkpoint with new name
import os
os.makedirs('/home/kenpeter/work/coder/checkpoints/san/', exist_ok=True)
torch.save(st_edited, '/home/kenpeter/work/coder/checkpoints/san/san_all_edited.pt')

# Save edit summary
with open('/home/kenpeter/work/coder/all_weight_edit_summary.json', 'w') as f:
    json.dump(edit_summary, f, indent=2)

print(f'EDITED ALL {edit_summary["edited_groups"]} WEIGHT GROUPS')
print(f'=' * 50)
print(f'Total groups in model: {edit_summary["total_groups"]}')
print(f'Groups successfully edited: {edit_summary["edited_groups"]}')
print(f'Edit amount: {edit_summary["edit_amount"]}')
print(f'Edited checkpoint saved: san_all_edited.pt')
print(f'Summary saved: all_weight_edit_summary.json')
print()
print('Per-group change stats (first 5 groups):')
for i, (key, stats) in enumerate(edit_summary['weight_changes'].items(), 1):
    if i <= 5:
        print(f'  {key}: mag={stats["change_magnitude"]:.8f}, max={stats["max_change"]:.8f}, '
              f'orig_mean={stats["original_mean"]:.6e}, edited_mean={stats["edited_mean"]:.6e}')
