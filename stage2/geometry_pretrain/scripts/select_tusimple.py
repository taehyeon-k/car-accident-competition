"""Select the TuSimple members to fetch: frames {12,17,19,20} of all train clips,
the same frames of a fixed random 600 test clips (seed 0), and all label JSONs."""

import random
import sys

FRAMES = {"12.jpg", "17.jpg", "19.jpg", "20.jpg"}


def main(listing="/workspace/data/geometry/tusimple/kaggle_members.txt", out="/workspace/data/geometry/tusimple/members_selected.txt"):
    members = [l.split()[0] for l in open(listing)]
    test_clips = sorted({m.rsplit("/", 1)[0] for m in members if m.startswith("TUSimple/test_set/clips/")})
    random.Random(0).shuffle(test_clips)
    test_keep = set(test_clips[:600])
    keep = []
    for m in members:
        if m.endswith("/"):
            continue
        clip, name = m.rsplit("/", 1)
        if m.startswith("TUSimple/train_set/clips/") and name in FRAMES:
            keep.append(m)
        elif m.startswith("TUSimple/test_set/clips/") and clip in test_keep and name in FRAMES:
            keep.append(m)
        elif m.endswith(".json") or m.endswith("readme.md"):
            keep.append(m)
    open(out, "w").write("\n".join(keep))
    print(len(keep))


if __name__ == "__main__":
    main(*sys.argv[1:])
