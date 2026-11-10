import sys
sys.path.insert(0, '/home/kenpeter/work/coder')
from infer import load_model

model, tok, st = load_model('/home/kenpeter/work/coder/checkpoints/san/san_latest.pt')
print(f'Model loaded successfully')
print(f'Step: {st.get("step")}')
print(f'Loss: {st.get("loss")}')

# Try a very simple generation
prompt = 'def add(a, b):\n    return a + b\n'
print(f'Prompt: {prompt}')
