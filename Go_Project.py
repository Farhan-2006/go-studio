"""19x19 Go move adviser. Run --help for reproducible training and evaluation.

Team members
Farhan Naik - 2410110422
Hunar Bhatia - 2410110427
Aniket Sharma - 2410110516
Atul Kumar Arya - 2410110088
"""
from pathlib import Path
import argparse, hashlib, json, time, tarfile, warnings, os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np
ROOT = Path(__file__).resolve().parent
N = 19
PASS = 361
SEED = 42
COLS = 'ABCDEFGHJKLMNOPQRST'
NEIGHBORS = [[q for q in (p-19 if p>=19 else -1, p+19 if p<342 else -1,
                         p-1 if p%19 else -1, p+1 if p%19<18 else -1) if q>=0] for p in range(361)]

def coordinate(p):
    return 'pass' if p==PASS else f'{COLS[p%19]}{19-p//19}'

def group_info(board):
    """Return group IDs, stone sets, and liberty sets; each group is visited once."""
    flat=np.asarray(board).reshape(-1); ids=np.full(361,-1,int); groups=[]; liberties=[]
    for p in np.flatnonzero(flat):
        if ids[p]>=0: continue
        gid=len(groups); stones={int(p)}; todo=[int(p)]; libs=set(); ids[p]=gid
        while todo:
            q=todo.pop()
            for z in NEIGHBORS[q]:
                if flat[z]==0: libs.add(z)
                elif flat[z]==flat[p] and ids[z]<0:
                    ids[z]=gid; stones.add(z); todo.append(z)
        groups.append(stones); liberties.append(libs)
    return ids,groups,liberties

def play(board, player, move, history=None):
    """No-suicide, positional-superko rules. Pass is exempt from repetition checks."""
    b=np.array(board,dtype=np.int8,copy=True).reshape(19,19)
    if player not in (-1,1): raise ValueError('Player must be 1 (Black) or -1 (White).')
    if move==PASS: return b
    if not 0<=move<361 or b.flat[move]: raise ValueError('Occupied or invalid intersection.')
    b.flat[move]=player
    ids,groups,libs=group_info(b)
    for gid in {ids[q] for q in NEIGHBORS[move] if b.flat[q]==-player}:
        if not libs[gid]:
            for q in groups[gid]: b.flat[q]=0
    ids,groups,libs=group_info(b)
    if not libs[ids[move]]: raise ValueError('Suicide move.')
    if history is not None and b.tobytes() in history: raise ValueError('Positional superko repetition.')
    return b

def validate_position(board):
    b=np.asarray(board)
    if b.shape!=(19,19) or not np.isin(b,[-1,0,1]).all(): raise ValueError('Expected 19x19 values -1, 0, 1.')
    if any(not x for x in group_info(b)[2]): raise ValueError('A stone group has no liberties; remove captured stones.')

FEATURE_NAMES = ['row_edge','col_edge','corner_distance','centre_distance','stage','pass',
                 'friendly_adjacent','enemy_adjacent','empty_adjacent','captured_stones',
                 'saved_atari_stones','joined_stones','result_liberties','self_atari',
                 'enemy_atari_groups','last_move_distance'] + [f'patch_{r}_{c}' for r in range(5) for c in range(5)]

def candidates(board, player, last=-1, history=None):
    """Compute legal moves and local candidate features without copying whole boards per move."""
    flat=np.asarray(board).reshape(-1); ids,groups,libs=group_info(board)
    moves=[]; features=[]; stage=np.count_nonzero(flat)/361
    padded=np.pad(np.asarray(board)*player,2,constant_values=2)
    for p in np.flatnonzero(flat==0):
        r,c=divmod(int(p),19); ns=NEIGHBORS[p]
        friendly={ids[q] for q in ns if flat[q]==player}; enemy={ids[q] for q in ns if flat[q]==-player}
        captured=set().union(*(groups[g] for g in enemy if len(libs[g])==1)) if enemy else set()
        own={int(p)}.union(*(groups[g] for g in friendly))
        newlibs={q for z in own for q in NEIGHBORS[z] if flat[q]==0 or q in captured}-{int(p)}
        if not newlibs: continue
        if history is not None:
            nb=flat.copy(); nb[p]=player
            for q in captured: nb[q]=0
            if nb.reshape(19,19).tobytes() in history: continue
        rowedge=min(r,18-r); coledge=min(c,18-c)
        vals=[rowedge/9,coledge/9,(rowedge+coledge)/18,(abs(r-9)+abs(c-9))/18,stage,0,
              sum(flat[q]==player for q in ns)/4,sum(flat[q]==-player for q in ns)/4,
              sum(flat[q]==0 for q in ns)/4,min(len(captured),20)/20,
              min(sum(len(groups[g]) for g in friendly if len(libs[g])==1),20)/20,
              min(len(own),20)/20,min(len(newlibs),20)/20,int(len(newlibs)==1),
              sum(len(libs[g]-{int(p)})==1 for g in enemy)/4,
              (abs(r-last//19)+abs(c-last%19))/36 if 0<=last<361 else 1]
        vals.extend(padded[r:r+5,c:c+5].reshape(-1))
        moves.append(int(p)); features.append(vals)
    vals=np.zeros(len(FEATURE_NAMES)); vals[4]=stage; vals[5]=1
    moves.append(PASS); features.append(vals)
    return np.asarray(moves),np.asarray(features,dtype=np.float32)

def planes(board,player,last=-1):
    b=np.asarray(board); x=np.zeros((5,19,19),np.float32)
    x[0]=b==player; x[1]=b==-player; x[2]=b==0
    if 0<=last<361: x[3].flat[last]=1
    x[4]=np.count_nonzero(b)/361
    return x

def make_cnn():
    import torch.nn as nn
    class PolicyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.body=nn.Sequential(nn.Conv2d(5,24,5,padding=2),nn.ReLU(),
                nn.Conv2d(24,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.ReLU())
            self.policy=nn.Conv2d(32,1,1); self.pass_head=nn.Linear(32,1)
        def forward(self,x):
            import torch
            h=self.body(x)
            return torch.cat([self.policy(h).flatten(1),self.pass_head(h.mean((2,3)))],1)
    return PolicyCNN()

def dump(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2),encoding='utf8')

def prepare(max_games=1200,per_game=16):
    """Parse archive in memory, keep only fully legal non-handicap 19x19 main lines."""
    from sgfmill import sgf
    (ROOT/'data/processed').mkdir(parents=True,exist_ok=True)
    archive=ROOT/'data/raw/games.tgz'
    if not archive.exists():
        import urllib.request
        archive.parent.mkdir(parents=True,exist_ok=True)
        urllib.request.urlretrieve('https://homepages.cwi.nl/~aeb/go/games/games.tgz',archive)
    rng=np.random.default_rng(SEED); seen=set(); records=[]; skipped={}; scanned=0
    with tarfile.open(archive,'r:gz') as tar:
        members=[m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith('.sgf')]
        rng.shuffle(members)
        for m in members:
            if len(records)>=max_games: break
            scanned+=1
            try:
                game=sgf.Sgf_game.from_bytes(tar.extractfile(m).read())
                if game.get_size()!=19: raise ValueError('board_size')
                nodes=list(game.get_main_sequence())
                if any(any(n.get_setup_stones()) for n in nodes): raise ValueError('setup_or_handicap')
                seq=[]
                for node in nodes:
                    color,pos=node.get_move()
                    if color is not None:
                        move=PASS if pos is None else (18-pos[0])*19+pos[1]
                        seq.append((1 if color=='b' else -1,move))
                if len(seq)<50: raise ValueError('short_game')
                # Canonical move digest prevents rotated/reflected duplicate game leakage.
                signatures=[]
                for reflect in (False,True):
                    for k in range(4):
                        transformed=[]
                        for col,p in seq:
                            r,c=divmod(p,19)
                            if p!=PASS:
                                if reflect: c=18-c
                                for _ in range(k): r,c=c,18-r
                                p=r*19+c
                            transformed.append((col,p))
                        signatures.append(str(transformed))
                gid=hashlib.sha256(min(signatures).encode()).hexdigest()
                if gid in seen: raise ValueError('duplicate')
                b=np.zeros((19,19),np.int8); hist={b.tobytes()}; snapshots=[]; last=-1
                choices=set(rng.choice(np.arange(4,len(seq)),min(per_game,len(seq)-4),replace=False).tolist())
                expected=1
                for i,(col,p) in enumerate(seq):
                    if col!=expected: raise ValueError('nonalternating')
                    if i in choices: snapshots.append((b.copy(),col,p,last))
                    b=play(b,col,p,hist); hist.add(b.tobytes()); last=p; expected=-col
                records.append((gid,m.name,snapshots)); seen.add(gid)
            except Exception as e:
                reason=str(e) if isinstance(e,ValueError) else type(e).__name__
                skipped[reason]=skipped.get(reason,0)+1
            if len(records) and len(records)%100==0: print('Games accepted:',len(records),flush=True)
    if len(records)<30: raise RuntimeError('Not enough valid games.')
    rng.shuffle(records); a=int(len(records)*.8); z=int(len(records)*.9)
    manifest={'source':'https://homepages.cwi.nl/~aeb/go/games/','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
              'seed':SEED,'scanned':scanned,'accepted_games':len(records),'skipped':skipped,'split':{},
              'rules':'19x19, no suicide, positional superko; no setup stones; alternating turns',
              'positions_per_game':per_game}
    example=[]
    for name,part in [('train',records[:a]),('validation',records[a:z]),('test',records[z:])]:
        bs=[]; ps=[]; ys=[]; ls=[]; gids=[]
        for gid,source,positions in part:
            for b,p,y,last in positions:
                bs.append(b); ps.append(p); ys.append(y); ls.append(last); gids.append(gid)
        np.savez_compressed(ROOT/f'data/processed/{name}.npz',boards=bs,players=ps,targets=ys,last=ls,game_ids=gids)
        manifest['split'][name]={'games':len(part),'positions':len(bs),'game_ids':[r[0] for r in part]}
        if name=='test':
            for j in np.linspace(0,len(bs)-1,8,dtype=int):
                example.append({'name':f'Test position {len(example)+1}', 'board':bs[j].tolist(),'player':int(ps[j]),
                                'last':int(ls[j]),'recorded_move':int(ys[j]),'game_id':gids[j],
                                'history_available':False})
    dump(ROOT/'data/manifest.json',manifest); dump(ROOT/'examples/positions.json',example)
    print({k:{a:b for a,b in v.items() if a!='game_ids'} for k,v in manifest['split'].items()},flush=True)

def load_split(name):
    return dict(np.load(ROOT/f'data/processed/{name}.npz'))

def train(epochs=12):
    import joblib, torch
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4); torch.manual_seed(SEED); rng=np.random.default_rng(SEED)
    trainset=load_split('train'); val=load_split('validation'); X=[]; y=[]
    start=time.perf_counter()
    for i,(b,p,t,last) in enumerate(zip(trainset['boards'],trainset['players'],trainset['targets'],trainset['last'])):
        moves,f=candidates(b,int(p),int(last)); positive=np.flatnonzero(moves==t)
        if not len(positive): continue
        neg=np.flatnonzero(moves!=t); take=rng.choice(neg,min(12,len(neg)),replace=False)
        selected=np.r_[positive,take]; X.extend(f[selected]); y.extend((moves[selected]==t).astype(int))
        if i%2000==0: print('Candidate features:',i,flush=True)
    X=np.asarray(X,np.float32); y=np.asarray(y); feature_seconds=time.perf_counter()-start
    models={
        'logistic_regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=300,C=1,class_weight='balanced',random_state=SEED)),
        'random_forest':RandomForestClassifier(n_estimators=100,max_depth=16,min_samples_leaf=8,class_weight='balanced',n_jobs=4,random_state=SEED),
        'mlp':make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(64,32),max_iter=45,batch_size=512,
                   early_stopping=False,random_state=SEED,learning_rate_init=.001))}
    metadata={'seed':SEED,'feature_names':FEATURE_NAMES,'negative_samples_per_positive':12,
              'feature_preparation_seconds':feature_seconds,'training_positions':len(trainset['boards']),
              'candidate_training_rows':len(X),'models':{},'note':'Candidate scores trained with sampled negatives are not calibrated move or win probabilities.'}
    ROOT.joinpath('models').mkdir(exist_ok=True)
    for name,model in models.items():
        print('Training',name,flush=True); start=time.perf_counter()
        with threadpool_limits(limits=4): model.fit(X,y)
        path=ROOT/f'models/{name}.joblib'; joblib.dump(model,path,compress=3)
        metadata['models'][name]={'training_seconds':time.perf_counter()-start,'bytes':path.stat().st_size,
                                 'parameters':str(model)}
        dump(ROOT/'models/training_metadata.json',metadata)
    net=make_cnn(); optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    xt=torch.from_numpy(np.stack([planes(b,int(p),int(l)) for b,p,l in zip(trainset['boards'],trainset['players'],trainset['last'])]))
    yt=torch.tensor(trainset['targets'],dtype=torch.long)
    xv=torch.from_numpy(np.stack([planes(b,int(p),int(l)) for b,p,l in zip(val['boards'],val['players'],val['last'])]))
    yv=torch.tensor(val['targets'],dtype=torch.long)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xt,yt),batch_size=128,shuffle=True)
    best=float('inf'); curves=[]; start=time.perf_counter()
    for epoch in range(epochs):
        net.train(); total=0
        for xb,yb in loader:
            optimizer.zero_grad(); logits=net(xb)
            # Occupancy mask is identical at training and inference; full legality applied in inference.
            occupied=xb[:,2].flatten(1)==0
            logits[:,:361]=logits[:,:361].masked_fill(occupied,-1e9)
            loss=torch.nn.functional.cross_entropy(logits,yb); loss.backward(); optimizer.step(); total+=loss.item()*len(xb)
        net.eval(); vl=0
        with torch.no_grad():
            for j in range(0,len(xv),128):
                out=net(xv[j:j+128]); out[:,:361]=out[:,:361].masked_fill(xv[j:j+128,2].flatten(1)==0,-1e9)
                vl+=torch.nn.functional.cross_entropy(out,yv[j:j+128],reduction='sum').item()
        vl/=len(xv); curves.append({'epoch':epoch+1,'train_loss':total/len(xt),'validation_loss':vl})
        if vl<best: best=vl; torch.save(net.state_dict(),ROOT/'models/cnn.pt')
        dump(ROOT/'results/learning_curve.json',curves)
        print(f'CNN epoch {epoch+1}/{epochs}: train {total/len(xt):.3f}, validation {vl:.3f}',flush=True)
    metadata['models']['cnn']={'training_seconds':time.perf_counter()-start,'bytes':(ROOT/'models/cnn.pt').stat().st_size,
        'epochs':epochs,'best_validation_loss':best,'selection':'lowest validation cross entropy','parameters':sum(p.numel() for p in net.parameters())}
    dump(ROOT/'models/training_metadata.json',metadata)

MODEL_LABELS = {
    'logistic_regression': 'Logistic regression',
    'random_forest': 'Random forest',
    'mlp': 'MLP',
    'cnn': 'CNN',
}


def combine_rankings(scores):
    """Average each model's percentile ranks, sharing ranks for tied scores."""
    from scipy.stats import rankdata

    move_count = scores.shape[1]
    if move_count == 1:
        return np.full(1, 100.0)
    ranks = rankdata(scores, axis=1, method='average')
    return (ranks.mean(axis=0) - 1) * 100 / (move_count - 1)


class Adviser:
    def __init__(self):
        self.models = {}

    def _load_model(self, model):
        if model not in MODEL_LABELS:
            raise ValueError(f'Unknown model: {model}')
        if model in self.models:
            return
        if model == 'cnn':
            import torch

            torch.set_num_threads(4)
            net = make_cnn()
            weights = torch.load(ROOT / 'models/cnn.pt', map_location='cpu', weights_only=True)
            net.load_state_dict(weights)
            net.eval()
            self.models[model] = net
        else:
            import joblib

            self.models[model] = joblib.load(ROOT / f'models/{model}.joblib')

    def _scores(self, model, board, player, last, moves, features):
        predictor = self.models[model]
        if model == 'cnn':
            import torch

            with torch.no_grad():
                inputs = torch.from_numpy(planes(board, player, last)[None])
                return predictor(inputs)[0].numpy()[moves]
        return predictor.predict_proba(features)[:, 1]

    def recommend(self, board, player, model='cnn', last=-1, history=None, k=5, include_breakdown=False):
        if player not in (-1, 1):
            raise ValueError('Choose Black or White to move.')
        validate_position(board)
        model_names = tuple(MODEL_LABELS) if model == 'combined' else (model,)
        for name in model_names:
            self._load_model(name)
        start = time.perf_counter()
        moves, features = candidates(board, player, last, history)
        scores = np.stack([
            self._scores(name, board, player, last, moves, features)
            for name in model_names
        ])
        ranking_scores = combine_rankings(scores) if model == 'combined' else scores[0]
        def ranked(values):
            order = np.argsort(-values, kind='stable')[:k]
            return [
                {'move': int(moves[i]), 'coordinate': coordinate(int(moves[i])),
                 'score': float(values[i])}
                for i in order
            ]

        result = ranked(ranking_scores)
        if model == 'combined' and result:
            result[0]['model_choices'] = {
                name: coordinate(int(moves[np.argmax(values)]))
                for name, values in zip(model_names, scores)
            }
            if include_breakdown:
                result[0]['model_rankings'] = {
                    name: ranked(values) for name, values in zip(model_names, scores)
                }
        return result, time.perf_counter() - start


def evaluate_combined():
    """Compare the fixed equal-weight rule on the original held-out sample."""
    import csv

    results = ROOT / 'results'
    environment = json.loads((results / 'environment.json').read_text())
    indices = environment['test_sample_indices']
    ds = load_split('test')
    adviser = Adviser()
    top1, top5, latencies = [], [], []
    for index in indices:
        predictions, elapsed = adviser.recommend(
            ds['boards'][index], int(ds['players'][index]), 'combined', int(ds['last'][index])
        )
        target = int(ds['targets'][index])
        top1.append(predictions[0]['move'] == target)
        top5.append(any(item['move'] == target for item in predictions))
        latencies.append(elapsed * 1000)
    row = {'model': 'combined', 'positions': len(indices), 'top1': np.mean(top1),
           'top5': np.mean(top5), 'median_ms': np.median(latencies)}
    with (results / 'combined_metrics.csv').open('w', newline='', encoding='utf8') as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row)


def evaluate(limit=500):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (ROOT/'results/figures').mkdir(parents=True,exist_ok=True)
    ds=load_split('test'); rng=np.random.default_rng(SEED)
    indices=rng.choice(len(ds['boards']),min(limit,len(ds['boards'])),replace=False)
    adviser=Adviser(); rows=[]; predictions=[]
    meta=json.loads((ROOT/'models/training_metadata.json').read_text())
    for name in ['logistic_regression','random_forest','mlp','cnn']:
        one=[]; five=[]; secs=[]; bygame={}
        for n,i in enumerate(indices):
            top,elapsed=adviser.recommend(ds['boards'][i],int(ds['players'][i]),name,int(ds['last'][i]))
            hit=int(top[0]['move']==int(ds['targets'][i])); h5=int(int(ds['targets'][i]) in [x['move'] for x in top])
            one.append(hit); five.append(h5); secs.append(elapsed)
            gid=str(ds['game_ids'][i]); bygame.setdefault(gid,[]).append(hit)
            predictions.append({'model':name,'test_index':int(i),'game_id':gid,'target':int(ds['targets'][i]),'top':top,'seconds':elapsed})
        # Cluster bootstrap by game, preserving dependence of positions within games.
        groups=list(bygame.values()); boot=[]
        for _ in range(1000):
            chosen=rng.integers(len(groups),size=len(groups)); arr=[x for j in chosen for x in groups[j]]; boot.append(np.mean(arr))
        low,high=np.percentile(boot,[2.5,97.5])
        row={'model':name,'positions':len(indices),'top1':np.mean(one),'top5':np.mean(five),
             'top1_ci_low':low,'top1_ci_high':high,'median_ms':np.median(secs)*1000,
             'p95_ms':np.percentile(secs,95)*1000,**{k:v for k,v in meta['models'][name].items() if k in ['training_seconds','bytes']}}
        rows.append(row); print(row,flush=True)
    frame=pd.DataFrame(rows); frame.to_csv(ROOT/'results/metrics.csv',index=False)
    dump(ROOT/'results/predictions.json',predictions)
    fig,axes=plt.subplots(1,2,figsize=(12,4)); x=np.arange(4)
    axes[0].bar(x-.18,frame.top1,.36,label='Top 1'); axes[0].bar(x+.18,frame.top5,.36,label='Top 5')
    axes[0].set(ylabel='Agreement with recorded move',ylim=(0,1)); axes[0].legend()
    axes[1].bar(x,frame.median_ms); axes[1].set(ylabel='Median recommendation time (ms)')
    for ax in axes: ax.set_xticks(x,frame.model,rotation=20)
    fig.tight_layout(); fig.savefig(ROOT/'results/figures/comparison.png',dpi=160); plt.close(fig)
    import platform, importlib.metadata as md
    dump(ROOT/'results/environment.json',{'python':platform.python_version(),'platform':platform.platform(),
         'packages':{k:md.version(k) for k in ['numpy','scikit-learn','torch','sgfmill','matplotlib']},
         'seed':SEED,'test_sample_size':len(indices),'test_sample_indices':indices.tolist(),
         'latency_note':'Includes legal-move feature generation, excludes first model load. CPU, sequential evaluation.'})

def selftest():
    import unittest
    class RulesTests(unittest.TestCase):
        def test_empty(self):
            b=np.zeros((19,19),np.int8); m,f=candidates(b,1)
            self.assertEqual(len(m),362); self.assertEqual(f.shape,(362,len(FEATURE_NAMES)))
        def test_capture(self):
            b=np.zeros((19,19),np.int8); b[1,1]=-1
            b[0,1]=b[1,0]=b[2,1]=1
            self.assertEqual(play(b,1,21)[1,1],0)
        def test_suicide(self):
            b=np.zeros((19,19),np.int8); b[0,1]=b[1,0]=b[2,1]=b[1,2]=-1
            with self.assertRaises(ValueError): play(b,1,20)
            self.assertNotIn(20,candidates(b,1)[0])
        def test_superko(self):
            b=np.zeros((19,19),np.int8); nxt=play(b,1,0)
            with self.assertRaises(ValueError): play(b,1,0,{nxt.tobytes()})
            self.assertNotIn(0,candidates(b,1,history={nxt.tobytes()})[0])
            self.assertTrue(np.array_equal(play(b,1,PASS,{b.tobytes()}),b))
        def test_reference_rules(self):
            from sgfmill import boards
            rng=np.random.default_rng(9); b=np.zeros((19,19),np.int8); ref=boards.Board(19); player=1
            for _ in range(150):
                ms,_=candidates(b,player); move=int(rng.choice(ms[ms!=PASS])); b=play(b,player,move)
                r,c=divmod(move,19); ref.play(18-r,c,'b' if player==1 else 'w')
                for rr in range(19):
                    for cc in range(19):
                        value=ref.get(18-rr,cc); self.assertEqual(b[rr,cc],0 if value is None else 1 if value=='b' else -1)
                player=-player
        def test_feature_legality(self):
            rng=np.random.default_rng(2); b=np.zeros((19,19),np.int8)
            for turn in range(80):
                player=1 if turn%2==0 else -1; ms,_=candidates(b,player)
                for m in rng.choice(ms,min(8,len(ms)),replace=False): play(b,player,int(m))
                b=play(b,player,int(rng.choice(ms)))
        def test_coordinates(self):
            self.assertEqual(coordinate(0),'A19'); self.assertEqual(coordinate(360),'T1')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RulesTests))
    if not result.wasSuccessful(): raise SystemExit(1)
    if (ROOT/'data/manifest.json').exists():
        manifest=json.loads((ROOT/'data/manifest.json').read_text()); sets=[set(v['game_ids']) for v in manifest['split'].values()]
        assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
        print('Game split disjointness: PASS')

def gui():
    """Open the local move adviser with a ready-to-analyze example."""
    import csv
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title('Go Studio')
    root.geometry('1120x790')
    root.minsize(820, 670)
    background, panel, border = '#141a19', '#202825', '#34413a'
    ink, muted, accent = '#edf1eb', '#aebbb1', '#bce1aa'
    root.configure(bg=background)
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', font=('Segoe UI', 10), background=background, foreground=ink)
    style.configure('TFrame', background=background)
    style.configure('Panel.TFrame', background=panel)
    style.configure('Panel.TLabel', background=panel, foreground=ink)
    style.configure('Quiet.TLabel', background=panel, foreground=muted)
    style.configure('TButton', padding=(10, 6), background=panel, foreground=ink,
                    bordercolor=border, lightcolor=panel, darkcolor=panel)
    style.map('TButton', background=[('active', border)], foreground=[('disabled', '#78877f')])
    style.configure('Primary.TButton', background=accent, foreground='#16271b',
                    font=('Segoe UI', 11, 'bold'), padding=(12, 9))
    style.map('Primary.TButton', background=[('active', '#d1ecc3'), ('disabled', border)],
              foreground=[('disabled', muted)])
    style.configure('TRadiobutton', background=background, foreground=ink, padding=(3, 4),
                    indicatorbackground=panel)
    style.map('TRadiobutton', background=[('active', panel)], indicatorcolor=[('selected', accent)])
    style.configure('TCombobox', fieldbackground=panel, background=panel, foreground=ink,
                    arrowcolor=ink, bordercolor=border, padding=4)
    style.map('TCombobox', fieldbackground=[('readonly', panel)], foreground=[('readonly', ink)])
    root.option_add('*TCombobox*Listbox.background', panel)
    root.option_add('*TCombobox*Listbox.foreground', ink)
    root.option_add('*TCombobox*Listbox.selectBackground', border)
    style.configure('TNotebook', background=background, borderwidth=0)
    style.configure('TNotebook.Tab', padding=(16, 7), background=panel, foreground=muted)
    style.map('TNotebook.Tab', background=[('selected', border)], foreground=[('selected', ink)])
    style.configure('Treeview', background=panel, fieldbackground=panel, foreground=ink,
                    borderwidth=0, rowheight=25)
    style.configure('Treeview.Heading', background=border, foreground=ink, relief='flat')
    style.map('Treeview', background=[('selected', '#3d5844')], foreground=[('selected', ink)])

    adviser = Adviser()
    examples = json.loads((ROOT / 'examples/positions.json').read_text())
    state = {'board': np.zeros((19, 19), np.int8), 'history': None,
             'last': -1, 'top': [], 'undo': [], 'example': None, 'geometry': None}
    player = tk.StringVar(value='Black')
    brush = tk.StringVar(value='Black')
    example = tk.StringVar(value=examples[0]['name'])
    result = tk.StringVar(value='Ready to analyze')
    hint = tk.StringVar(value='Find a move using all four models.')
    detail_text = tk.StringVar(value='Find a move to compare the models.')
    method = tk.StringVar(value='Combined')
    methods = {'Combined': 'combined', **{label: name for name, label in MODEL_LABELS.items()}}
    board_title = tk.StringVar()

    header = ttk.Frame(root, padding=(24, 12, 24, 10))
    header.pack(fill='x')
    heading = ttk.Frame(header)
    heading.pack(side='left')
    ttk.Label(heading, text='Go Studio', font=('Georgia', 26, 'bold')).pack(anchor='w')
    ttk.Label(heading, text='19 × 19 move adviser', foreground=muted).pack(anchor='w')
    example_picker = ttk.Combobox(header, textvariable=example, state='readonly',
                                 values=[item['name'] for item in examples], width=17)
    example_picker.pack(side='right')
    ttk.Label(header, text='Example', foreground=muted).pack(side='right', padx=(0, 9))

    content = ttk.Frame(root, padding=(24, 0, 24, 16))
    content.pack(fill='both', expand=True)
    content.columnconfigure(0, weight=1)
    content.rowconfigure(0, weight=1)
    board_area = ttk.Frame(content)
    board_area.grid(row=0, column=0, sticky='nsew', padx=(0, 20))
    ttk.Label(board_area, textvariable=board_title, foreground=muted).pack(anchor='w', pady=(0, 5))
    canvas = tk.Canvas(board_area, bg=background, highlightthickness=0)
    canvas.pack(fill='both', expand=True)

    side = ttk.Frame(content, padding=(0, 0, 0, 0))
    side.grid(row=0, column=1, rowspan=2, sticky='nsew')
    tabs = ttk.Notebook(side, width=305)
    tabs.pack(fill='x')
    play_panel = ttk.Frame(tabs, style='Panel.TFrame', padding=14)
    analysis_panel = ttk.Frame(tabs, style='Panel.TFrame', padding=12)
    tabs.add(play_panel, text='Play')
    tabs.add(analysis_panel, text='Analysis')

    ttk.Label(play_panel, text='Who moves next?', style='Panel.TLabel',
              font=('Segoe UI', 11, 'bold')).pack(anchor='w')
    ttk.Combobox(play_panel, textvariable=player, values=['Black', 'White'],
                 state='readonly').pack(fill='x', pady=(5, 10))
    analyze_button = ttk.Button(play_panel, text='Find next move', command=lambda: analyze(),
                                style='Primary.TButton')
    analyze_button.pack(fill='x')
    ttk.Label(play_panel, textvariable=result, style='Panel.TLabel', font=('Georgia', 19, 'bold'),
              wraplength=265).pack(anchor='w', pady=(16, 0))
    ttk.Label(play_panel, textvariable=hint, style='Quiet.TLabel', wraplength=265,
              justify='left').pack(anchor='w', pady=(5, 12))
    apply_button = ttk.Button(play_panel, text='Play this move', command=lambda: apply_move(),
                              state='disabled')
    apply_button.pack(fill='x')
    details_button = ttk.Button(play_panel, text='Model breakdown',
                                command=lambda: tabs.select(analysis_panel), state='disabled')
    details_button.pack(fill='x', pady=(6, 0))

    compare_button = ttk.Button(analysis_panel, text='Analyze position', command=lambda: analyze(),
                                style='Primary.TButton')
    compare_button.pack(fill='x', pady=(0, 8))
    ttk.Label(analysis_panel, text='First choice by model', style='Panel.TLabel',
              font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 4))
    comparison = ttk.Treeview(analysis_panel, columns=('method', 'move'), show='headings',
                              height=5, selectmode='browse')
    comparison.heading('method', text='Method')
    comparison.heading('move', text='Move')
    comparison.column('method', width=175, stretch=True)
    comparison.column('move', width=70, anchor='center', stretch=False)
    comparison.pack(fill='x', pady=(0, 8))
    ttk.Label(analysis_panel, text='Ranked moves', style='Panel.TLabel',
              font=('Segoe UI', 10, 'bold')).pack(anchor='w')
    method_picker = ttk.Combobox(analysis_panel, textvariable=method, values=list(methods),
                                state='readonly')
    method_picker.pack(fill='x', pady=(4, 6))
    candidates_table = ttk.Treeview(analysis_panel, columns=('rank', 'move', 'score'),
                                    show='headings', height=5, selectmode='browse')
    for col, title, width in [('rank', 'Rank', 55), ('move', 'Move', 80), ('score', 'Score', 100)]:
        candidates_table.heading(col, text=title)
        candidates_table.column(col, width=width, anchor='center')
    candidates_table.pack(fill='x')
    ttk.Label(analysis_panel, textvariable=detail_text, style='Quiet.TLabel', wraplength=275,
              justify='left').pack(anchor='w', pady=(8, 0))

    editing = ttk.Frame(content)
    editing.grid(row=1, column=0, sticky='ew', padx=(0, 20), pady=(6, 0))
    ttk.Label(editing, text='Edit', foreground=muted).pack(side='left', padx=(0, 6))
    for name in ('Black', 'White', 'Erase'):
        ttk.Radiobutton(editing, text=name, variable=brush, value=name).pack(side='left')
    ttk.Button(editing, text='Undo', command=lambda: undo()).pack(side='right')
    ttk.Button(editing, text='Clear', command=lambda: clear()).pack(side='right', padx=5)

    def draw(_event=None):
        canvas.delete('all')
        width, height = canvas.winfo_width(), canvas.winfo_height()
        size = min(width, height) - 10
        step = (size - 62) / 18
        if step <= 0:
            return
        left, top = (width-size)/2, (height-size)/2
        x0, y0 = left+31, top+31
        state['geometry'] = (x0, y0, step)
        canvas.create_rectangle(left+4, top+5, left+size+4, top+size+5, fill='#0b110f', outline='')
        canvas.create_rectangle(left, top, left+size, top+size, fill='#dcb781', outline='#f2d4a6')
        canvas.create_rectangle(left+5, top+5, left+size-5, top+size-5, outline='#b58b5c')
        for i in range(19):
            x, y = x0+i*step, y0+i*step
            canvas.create_line(x0, y, x0+18*step, y, fill='#775735')
            canvas.create_line(x, y0, x, y0+18*step, fill='#775735')
            canvas.create_text(x, y0-15, text=COLS[i], fill='#634524', font=('Segoe UI', 9))
            canvas.create_text(x0-16, y, text=str(19-i), fill='#634524', font=('Segoe UI', 9))
        for row in (3, 9, 15):
            for col in (3, 9, 15):
                x, y = x0+col*step, y0+row*step
                canvas.create_oval(x-2.8, y-2.8, x+2.8, y+2.8, fill='#684727', outline='')
        for move in np.flatnonzero(state['board']):
            row, col = divmod(int(move), 19)
            x, y = x0+col*step, y0+row*step
            radius = step*.455
            black = state['board'].flat[move] == 1
            canvas.create_oval(x-radius+1, y-radius+2, x+radius+1, y+radius+2, fill='#a48055', outline='')
            canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                               fill='#242927' if black else '#f9f3e5', outline='#171d1a' if black else '#b19a79')
            canvas.create_arc(x-radius+2, y-radius+2, x+radius-2, y+radius-2, start=65, extent=105,
                              style='arc', outline='#505651' if black else '#ffffff')
            if move == state['last']:
                canvas.create_oval(x-2, y-2, x+2, y+2, fill='#be8260', outline='')
        if state['top']:
            in_analysis = tabs.select() == str(analysis_panel)
            ranking = selected_ranking() if in_analysis else state['top'][:1]
            selected = candidates_table.selection() if in_analysis else ()
            for rank, item in enumerate(ranking, 1):
                if item['move'] == PASS:
                    continue
                row, col = divmod(item['move'], 19)
                x, y = x0+col*step, y0+row*step
                radius = step*.39
                highlighted = str(rank) in selected
                fill = '#274e39' if rank == 1 or highlighted else '#61806a'
                canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                                   fill=fill, outline='#e9f2de', width=2)
                if in_analysis:
                    canvas.create_text(x, y, text=str(rank), fill='white', font=('Segoe UI', 9, 'bold'))

    def snapshot():
        history = state['history']
        state['undo'].append((state['board'].copy(), state['last'],
                              None if history is None else history.copy(),
                              player.get(), state['example']))

    def invalidate(*_args):
        state['top'] = []
        result.set('Ready to analyze')
        hint.set('Find a move using all four models.')
        board_title.set(f'{player.get()} to move')
        update_analysis()
        apply_button.state(['disabled'])
        details_button.state(['disabled'])
        draw()

    def load_example(_event=None, remember=True):
        item = next(item for item in examples if item['name'] == example.get())
        if remember:
            snapshot()
        state.update(board=np.array(item['board'], dtype=np.int8),
                     last=item['last'], history=None, example=item)
        player.set('Black' if item['player'] == 1 else 'White')

    def clear():
        snapshot()
        example.set('')
        state.update(board=np.zeros((19, 19), np.int8), last=-1,
                     history={bytes(361)}, example=None)
        player.set('Black')

    def undo():
        if state['undo']:
            board, last, history, color, source = state['undo'].pop()
            state.update(board=board, last=last, history=history, example=source)
            example.set(source['name'] if source else '')
            player.set(color)

    def click(event):
        if state['geometry'] is None:
            return
        x, y, step = state['geometry']
        col, row = round((event.x-x)/step), round((event.y-y)/step)
        if 0 <= row < 19 and 0 <= col < 19:
            snapshot()
            state['board'][row, col] = {'Black': 1, 'White': -1, 'Erase': 0}[brush.get()]
            state.update(last=-1, history=None, example=None)
            example.set('')
            invalidate()

    def analyze():
        analyze_button.state(['disabled'])
        compare_button.state(['disabled'])
        apply_button.state(['disabled'])
        details_button.state(['disabled'])
        result.set('Finding a move…')
        hint.set('Reading the position and scoring legal moves.')
        root.after(25, recommend)

    def recommend():
        try:
            color = 1 if player.get() == 'Black' else -1
            top, _ = adviser.recommend(state['board'], color, 'combined',
                                       state['last'], state['history'], include_breakdown=True)
            state['top'] = top
            update_analysis()
            result.set(f'{player.get()} should play {top[0]["coordinate"]}')
            alternatives = ', '.join(item['coordinate'] for item in top[1:3])
            hint.set(f'Other options: {alternatives}' if alternatives else 'Pass is the only legal move.')
            apply_button.state(['!disabled'])
            details_button.state(['!disabled'])
            draw()
        except Exception as error:
            invalidate()
            result.set('Could not analyze this board')
            hint.set(str(error))
        finally:
            analyze_button.state(['!disabled'])
            compare_button.state(['!disabled'])

    def apply_move():
        if state['top']:
            color = 1 if player.get() == 'Black' else -1
            move = state['top'][0]['move']
            try:
                board = play(state['board'], color, move, state['history'])
            except ValueError as error:
                messagebox.showerror('Could not play move', str(error), parent=root)
                return
            snapshot()
            state.update(board=board, last=move, example=None)
            if state['history'] is not None:
                state['history'].add(board.tobytes())
            example.set('')
            player.set('White' if color == 1 else 'Black')

    def selected_ranking():
        if not state['top']:
            return []
        name = methods[method.get()]
        if name == 'combined':
            return state['top']
        return state['top'][0]['model_rankings'][name]

    def update_ranking(_event=None):
        candidates_table.delete(*candidates_table.get_children())
        combined = method.get() == 'Combined'
        candidates_table.heading('score', text='Rank score' if combined else 'Model score')
        for rank, item in enumerate(selected_ranking(), 1):
            score = f'{item["score"]:.1f}/100' if combined else f'{item["score"]:.3f}'
            candidates_table.insert('', 'end', iid=str(rank), values=(rank, item['coordinate'], score))
        draw()

    def update_analysis():
        comparison.delete(*comparison.get_children())
        if not state['top']:
            detail_text.set('Analyze this position to see each model’s ranked moves.')
        else:
            top = state['top'][0]
            comparison.insert('', 'end', iid='combined', values=('Combined', top['coordinate']))
            for name, choice in top['model_choices'].items():
                comparison.insert('', 'end', iid=name, values=(MODEL_LABELS[name], choice))
            color = 1 if player.get() == 'Black' else -1
            source = state['example']
            description = 'Combined averages the four models’ ranks. Individual scores use each model’s scale.'
            if source is not None and source['player'] == color:
                description = f'Recorded move: {coordinate(source["recorded_move"])}. ' + description
            if state['history'] is None:
                description += ' Repetition history unavailable.'
            detail_text.set(description)
        update_ranking()

    def choose_model(_event=None):
        selection = comparison.selection()
        if selection:
            method.set(MODEL_LABELS.get(selection[0], 'Combined'))
            update_ranking()

    def save_position():
        name = filedialog.asksaveasfilename(parent=root, defaultextension='.json',
                                            filetypes=[('Go position', '*.json')])
        if name:
            history = state['history']
            dump(name, {'board': state['board'].tolist(), 'player': 1 if player.get() == 'Black' else -1,
                        'last': state['last'], 'history': None if history is None else [h.hex() for h in history]})

    def load_position():
        name = filedialog.askopenfilename(parent=root, filetypes=[('Go position', '*.json')])
        if not name:
            return
        try:
            item = json.loads(Path(name).read_text(encoding='utf8'))
            board = np.asarray(item['board'])
            validate_position(board)
            if item['player'] not in (-1, 1):
                raise ValueError('Choose Black or White to move.')
            history = None if item.get('history') is None else {bytes.fromhex(h) for h in item['history']}
            if history is not None and any(len(h) != 361 for h in history):
                raise ValueError('Invalid position history.')
            last = int(item.get('last', -1))
            snapshot()
            state.update(board=board.astype(np.int8), last=last, history=history, example=None)
            example.set('')
            player.set('Black' if item['player'] == 1 else 'White')
        except (OSError, TypeError, KeyError, ValueError) as error:
            messagebox.showerror('Could not load position', str(error), parent=root)

    def show_results():
        rows = []
        for filename in ('metrics.csv', 'combined_metrics.csv'):
            path = ROOT / 'results' / filename
            if path.exists():
                with path.open(newline='', encoding='utf8') as file:
                    rows.extend(csv.DictReader(file))
        window = tk.Toplevel(root)
        window.title('Test results')
        table = ttk.Treeview(window, columns=('model', 'first', 'five'), show='headings', height=len(rows))
        for column, title in zip(('model', 'first', 'five'), ('Method', 'Top 1', 'Top 5')):
            table.heading(column, text=title)
            table.column(column, width=140)
        for row in rows:
            table.insert('', 'end', values=(MODEL_LABELS.get(row['model'], 'Combined'),
                         f'{float(row["top1"]):.1%}', f'{float(row["top5"]):.1%}'))
        table.pack(padx=14, pady=14)
        ttk.Label(window, text='Agreement with recorded moves on held-out positions.').pack(pady=(0, 14))

    menu = tk.Menu(root)
    file_menu = tk.Menu(menu, tearoff=False)
    file_menu.add_command(label='Open position…', command=load_position)
    file_menu.add_command(label='Save position…', command=save_position)
    menu.add_cascade(label='File', menu=file_menu)
    menu.add_command(label='Test results', command=show_results)
    root.configure(menu=menu)
    tabs.bind('<<NotebookTabChanged>>', draw)
    comparison.bind('<<TreeviewSelect>>', choose_model)
    method_picker.bind('<<ComboboxSelected>>', update_ranking)
    candidates_table.bind('<<TreeviewSelect>>', draw)
    canvas.bind('<Configure>', draw)
    canvas.bind('<Button-1>', click)
    example_picker.bind('<<ComboboxSelected>>', load_example)
    player.trace_add('write', invalidate)
    load_example(remember=False)
    root.mainloop()

def engine_reference(visits=64):
    """Offline KataGo reference for held-out examples; no use during training."""
    import subprocess
    engine_dir=ROOT/'reference_engine'
    config=engine_dir/'project_analysis.cfg'
    config.write_text('numAnalysisThreads = 1\nnumSearchThreads = 4\nnumEigenThreadsPerModel = 2\nmaxVisits = 64\nreportAnalysisWinratesAs = SIDETOMOVE\n',encoding='utf8')
    networks=list(engine_dir.glob('network*.gz'))
    if not networks: raise FileNotFoundError('Download an official KataGo network first.')
    examples=json.loads((ROOT/'examples/positions.json').read_text())
    adviser=Adviser(); queries=[]; output=[]
    for i,x in enumerate(examples):
        b=np.array(x['board'],np.int8)
        stones=[['B' if b.flat[p]==1 else 'W',coordinate(int(p))] for p in np.flatnonzero(b)]
        queries.append({'id':str(i),'initialStones':stones,'initialPlayer':'B' if x['player']==1 else 'W',
            'moves':[],'rules':'chinese','komi':7.5,'boardXSize':19,'boardYSize':19,'maxVisits':visits})
    start=time.perf_counter()
    proc=subprocess.run([str(engine_dir/'katago.exe'),'analysis','-config',str(config),'-model',str(networks[0])],
        input='\n'.join(json.dumps(q) for q in queries)+'\n',capture_output=True,text=True,timeout=900)
    (ROOT/'results/engine_stderr.txt').write_text(proc.stderr,encoding='utf8')
    if proc.returncode: raise RuntimeError(proc.stderr[-4000:])
    for line in proc.stdout.splitlines():
        try: result=json.loads(line)
        except json.JSONDecodeError: continue
        if 'moveInfos' not in result: continue
        x=examples[int(result['id'])]; b=np.array(x['board'],np.int8); comparisons={}
        engine_moves=[m['move'] for m in sorted(result['moveInfos'],key=lambda m:m['order'])]
        for name in ['logistic_regression','random_forest','mlp','cnn']:
            top,_=adviser.recommend(b,x['player'],name,x['last'])
            move=top[0]['coordinate']; comparisons[name]={'move':move,'matches_engine_top1':move==engine_moves[0],
                'in_engine_top5':move in engine_moves[:5]}
        output.append({'name':x['name'],'query':queries[int(result['id'])],'reference':result,'models':comparisons})
    if len(output)!=len(examples): raise RuntimeError(f'Only {len(output)} engine results; see engine_stderr.txt')
    dump(ROOT/'results/engine_reference.json',{'engine':'KataGo v1.16.3 Eigen AVX2 CPU',
        'network_source':(engine_dir/'network_source.txt').read_text(),'visits_per_position':visits,
        'seconds':time.perf_counter()-start,'limitations':'Small older network, limited search; recommendation is not ground truth. Standalone positions have no game history. Chinese rules, komi 7.5 assumed.',
        'positions':output})
    print('Engine reference complete:',len(output),'positions',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train','evaluate','test','demo','reference','recommend','evaluate-combined'],nargs='?',default='demo')
    parser.add_argument('--games',type=int,default=1200); parser.add_argument('--positions',type=int,default=16)
    parser.add_argument('--position',type=Path,help='JSON position with board, player, optional last and history')
    parser.add_argument('--model',choices=[*MODEL_LABELS, 'combined'],default='cnn')
    parser.add_argument('--epochs',type=int,default=12); parser.add_argument('--limit',type=int,default=500)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.games,args.positions)
    elif args.command=='train': train(args.epochs)
    elif args.command=='evaluate': evaluate(args.limit)
    elif args.command=='evaluate-combined': evaluate_combined()
    elif args.command=='test': selftest()
    elif args.command=='reference': engine_reference()
    elif args.command=='recommend':
        if args.position is None: parser.error('recommend requires --position FILE.json')
        x=json.loads(args.position.read_text()); validate_position(np.asarray(x['board']))
        history=None if x.get('history') is None else {bytes.fromhex(v) for v in x['history']}
        top,seconds=Adviser().recommend(np.array(x['board'],np.int8),x['player'],args.model,x.get('last',-1),history)
        print(json.dumps({'recommendations':top,'seconds':seconds,'history_available':history is not None},indent=2))
    else: gui()
if __name__=='__main__': main()
