"""Index the full v0.02 archive, define the assumed member pool, and check that the test set is disjoint from it.

Member pool = clips of the 35 words that are in neither validation_list.txt nor testing_list.txt (the standard
training split). Words other than the 10 keywords map to 'unknown'. Silence is not in the pool (see README).

Overlap checks against the official test set (data/test_set), all written to results/raw/overlap_check.json:
  1. same (word, file) name in the training split
  2. same speaker id (the part of the file name before '_nohash_') in the training split
  3. byte-identical audio (md5 of the 16000-sample int16 vector) in the training split
"""
import os, json, hashlib, collections
import numpy as np, pandas as pd
from common import *

SC = os.path.join(DATA, 'sc_v2'); TEST = os.path.join(DATA, 'test_set')
val = set(open(os.path.join(SC, 'validation_list.txt')).read().split())
tst = set(open(os.path.join(SC, 'testing_list.txt')).read().split())
words = sorted(d for d in os.listdir(SC) if os.path.isdir(os.path.join(SC, d)) and not d.startswith('_'))
print(len(words), 'words')

rows = []
for w in words:
    for f in sorted(os.listdir(os.path.join(SC, w))):
        if not f.endswith('.wav'): continue
        rel = f'{w}/{f}'
        split = 'val' if rel in val else 'test' if rel in tst else 'train'
        rows.append((rel, w, f.split('_')[0], split, w if w in KW else 'unknown'))
df = pd.DataFrame(rows, columns=['rel', 'word', 'speaker', 'split', 'cls'])
df['label'] = df['cls'].map(lambda c: FOLDER_TO_CLASS[c if c in KW else '_unknown_'])
print(df.split.value_counts().to_dict())
tr = df[df.split == 'train'].reset_index(drop=True)
tr[['rel', 'word', 'speaker', 'label']].to_csv(os.path.join(CACHE, 'train_pool.csv'), index=False)
print('train pool per label', tr.label.value_counts().sort_index().to_dict())

# test set index. Unknown-class files are renamed '<word>_<orig name>' and some names end in .wav.wav
trows = []
for folder in sorted(FOLDER_TO_CLASS):
    for f in sorted(os.listdir(os.path.join(TEST, folder))):
        if not f.endswith('.wav'): continue
        orig = f[:-4] if f.endswith('.wav.wav') else f
        if folder == '_unknown_': word, orig = orig.split('_', 1)
        else: word = folder
        trows.append((folder, f, word, orig, orig.split('_')[0], FOLDER_TO_CLASS[folder]))
te = pd.DataFrame(trows, columns=['folder', 'file', 'word', 'orig', 'speaker', 'label'])
te.to_csv(os.path.join(CACHE, 'test_index.csv'), index=False)
nosil = te[te.label != SILENCE]
print('test clips', len(te), 'non-silence', len(nosil))

# 1. name overlap
tr_names = set(tr.rel)
all_names = collections.defaultdict(set)
for r in df.itertuples(): all_names[r.split].add(r.rel)
nm = [f'{r.word}/{r.orig}' for r in nosil.itertuples()]
name_hits = {s: sum(n in all_names[s] for n in nm) for s in ('train', 'val', 'test')}
# 2. speakers
tr_spk = set(tr.speaker); te_spk = set(nosil.speaker)
spk_overlap = sorted(tr_spk & te_spk)
# 3. audio
def h(path): return hashlib.md5(read_wav16k(path).tobytes()).hexdigest()
tr_hash = {h(os.path.join(SC, r)): r for r in tr.rel}
te_hash = [h(os.path.join(TEST, r.folder, r.file)) for r in nosil.itertuples()]
audio_hits = sum(x in tr_hash for x in te_hash)
res = dict(test_clips_non_silence=len(nosil), test_speakers=len(te_spk), train_speakers=len(tr_spk),
           test_names_found_in={'train_split': name_hits['train'], 'validation_list': name_hits['val'], 'testing_list': name_hits['test']},
           speaker_ids_in_both=len(spk_overlap), speaker_ids_in_both_examples=spk_overlap[:10],
           test_clips_with_identical_audio_in_train=audio_hits)
json.dump(res, open(os.path.join(RAW, 'overlap_check.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
