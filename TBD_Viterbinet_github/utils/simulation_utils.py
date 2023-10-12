def Find_Accuracy(track,label,name=None):
    acc = 0.0
    track_len = len(label)
    spm = 0.0
    dist = 0.0
    for k in range(track_len):
        r = label[k][0][0].item()
        v = label[k][1][0].item()
        if (r,v) == track[k]:
            acc += 1
        dr = abs(r-track[k][0])
        dv = abs(v-track[k][1])
        dist += dr + dv
        if max(dr,dv) == 1:
            spm += 1
    acc /= track_len
    spm /= track_len
    dist /= track_len
    if name is not None:
        print(f'{name} accuracy =  {acc*100:.2f}%')
        print(f'{name} soft accuracy =  {(spm + acc) * 100:.2f}%')
        print(f'{name} average distance =  {dist:.2f} pixels')
    return acc, acc + spm, dist
