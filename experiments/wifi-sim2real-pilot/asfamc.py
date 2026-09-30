"""Minimal CMU ASF/AMC parser + forward kinematics -> joint positions in metres (z-up)."""
import numpy as np

UNIT = (1.0 / 0.45) * 2.54 / 100.0  # CMU length unit -> metres


def _rot(axis, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _euler(angles, order):
    # order like "XYZ": apply X first, then Y, then Z -> R = Rz @ Ry @ Rx
    R = np.eye(3)
    for ax, ang in zip(order.lower(), angles):
        R = _rot(ax, ang) @ R
    return R


def parse_asf(path):
    lines = [l.strip() for l in open(path, encoding="latin1")]
    bones = {"root": dict(dir=np.zeros(3), length=0.0, axis=np.zeros(3), dof=[], parent=None)}
    i = 0
    while i < len(lines):
        if lines[i].startswith(":bonedata"):
            i += 1
            while not lines[i].startswith(":hierarchy"):
                if lines[i] == "begin":
                    b = dict(dof=[])
                    i += 1
                    while lines[i] != "end":
                        t = lines[i].split()
                        if t[0] == "name":
                            name = t[1]
                        elif t[0] == "direction":
                            b["dir"] = np.array(list(map(float, t[1:4])))
                        elif t[0] == "length":
                            b["length"] = float(t[1]) * UNIT
                        elif t[0] == "axis":
                            b["axis"] = np.array(list(map(float, t[1:4])))
                        elif t[0] == "dof":
                            b["dof"] = [d.lower() for d in t[1:]]
                        i += 1
                    bones[name] = b
                i += 1
            i += 1
            while lines[i] != "end":
                if lines[i] == "begin":
                    i += 1
                    continue
                t = lines[i].split()
                for c in t[1:]:
                    bones[c]["parent"] = t[0]
                i += 1
        i += 1
    order = []  # topological order
    seen = {"root"}
    order.append("root")
    while len(order) < len(bones):
        for n, b in bones.items():
            if n not in seen and b.get("parent") in seen:
                order.append(n)
                seen.add(n)
    for n in order:
        b = bones[n]
        C = _euler(b["axis"], "XYZ")
        b["C"], b["Cinv"] = C, C.T
    return bones, order


def parse_amc(path):
    frames, cur = [], None
    for l in open(path, encoding="latin1"):
        l = l.strip()
        if not l or l.startswith("#") or l.startswith(":"):
            continue
        t = l.split()
        if t[0].isdigit() and len(t) == 1:
            cur = {}
            frames.append(cur)
        elif cur is not None:
            cur[t[0]] = np.array(list(map(float, t[1:])))
    return frames


def fk(bones, order, frames):
    """Return joints (T, J, 3) metres, z-up, and list of (parent_idx, child_idx) segments."""
    names = order
    idx = {n: i for i, n in enumerate(names)}
    T, J = len(frames), len(names)
    pos = np.zeros((T, J, 3))
    for t, fr in enumerate(frames):
        G, P = {}, {}
        for n in names:
            b = bones[n]
            if n == "root":
                v = fr["root"]
                P[n] = v[:3] * UNIT
                G[n] = _euler(v[3:6], "XYZ")
                pos[t, idx[n]] = P[n]
                continue
            vals = fr.get(n, np.zeros(len(b["dof"])))
            ang = {d: a for d, a in zip(b["dof"], vals)}
            M = _euler([ang.get("rx", 0), ang.get("ry", 0), ang.get("rz", 0)], "XYZ")
            L = b["C"] @ M @ b["Cinv"]
            G[n] = G[b["parent"]] @ L
            P[n] = P[b["parent"]] + G[n] @ (b["dir"] * b["length"])
            pos[t, idx[n]] = P[n]
    # CMU is y-up: convert to z-up (x, -z, y)
    pos = np.stack([pos[..., 0], -pos[..., 2], pos[..., 1]], -1)
    segs = [(idx[bones[n]["parent"]], idx[n]) for n in names if n != "root"]
    return pos, names, segs
