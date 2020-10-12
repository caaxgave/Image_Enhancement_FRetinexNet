import os

with open('./eval15/evel50.txt') as pf:
    fs = pf.read().split('\n')

for f in fs:
    for lh in ['low', 'high']:
        if os.path.exists(f_low):
            os.rename('./our485/%s/%s' % (lh, f), './eval15/%s/%s' % (lh, f))
