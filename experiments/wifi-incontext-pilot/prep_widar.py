import glob, os, numpy as np, multiprocessing as mp
SIX = ['1-Push&Pull', '2-Sweep', '3-Clap', '4-Slide', '6-Draw-O(H)', '9-Draw-Zigzag(H)']
fs = sorted(f for f in glob.glob('data/Widardata/*/*/*.csv') if f.split('/')[3] in SIX)
def load(f):
    a = np.loadtxt(f, delimiter=',', dtype=np.float32)
    p = os.path.basename(f).split('-')
    return a, SIX.index(f.split('/')[3]), int(p[0][4:]), int(p[2]), int(p[3])
if __name__ == '__main__':
    with mp.Pool(4) as pool: out = pool.map(load, fs, chunksize=200)
    ok = [o for o in out if o[0].shape == (22, 400)]
    print(len(fs), 'files,', len(ok), 'with shape (22,400)')
    X = np.stack([o[0] for o in ok]); y = np.array([o[1] for o in ok])
    user, loc, ori = (np.array([o[k] for o in ok]) for k in (2, 3, 4))
    np.savez('real/widar6.npz', X=X, y=y, user=user, loc=loc, ori=ori)
    print(X.shape, np.bincount(y), 'users', np.unique(user), 'locs', np.unique(loc), 'oris', np.unique(ori))
