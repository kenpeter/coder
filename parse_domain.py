import re

# Read the per-domain line from training log
with open('/home/kenpeter/work/coder/train_san.log', 'r') as f:
    lines = f.readlines()

# Find the last per-domain line
per_line = None
for line in reversed(lines):
    if 'per-domain' in line:
        per_line = line.strip()
        break

if not per_line:
    print("step=N/A coverage=N/A loss=N/A easy=N/A mid=N/A hard=N/A gold=N/A")
    import sys
    sys.exit(0)

print(f"step=N/A coverage=N/A loss=N/A easy=N/A mid=N/A hard=N/A gold=N/A")  # default

# Use regex to find all "code_XXX NUMBER" patterns
pattern = r'code_\w+\s+([0-9]*\.?[0-9]*)'
matches = re.findall(pattern, per_line)

print(f"step=N/A coverage=N/A loss=N/A easy=N/A mid=N/A hard=N/A gold=N/A")  # default

# Actually, let me just print the values we need
# The order in the log is typically: code_easy, code_gold, code_hard, code_mid
# But can vary. Let's just map based on what we find.

# Find all code_XXX patterns with their values
domain_pattern = r'code_(\w+)\s+([0-9]*\.?[0-9]*)'
domains = re.findall(domain_pattern, per_line)

print(f"Found {len(domains)} domains:")
for domain, value in domains:
    print(f"  {domain}: {value}")

# Map to our variables based on domain name
easy = gold = hard = mid = "N/A"
for domain, value in domains:
    if domain == "easy":
        easy = value
    elif domain == "gold":
        gold = value
    elif domain == "hard":
        hard = value
    elif domain == "mid":
        mid = value

print(f"easy={easy} mid={mid} hard={hard} gold={gold}")
