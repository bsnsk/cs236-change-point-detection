import matplotlib  # a workaround for virtualenv on macOS
matplotlib.use('TkAgg')  # a workaround for virtualenv on macOS
import matplotlib.pyplot as plt


# WARNING: Deprecated
def draw_roc(y_hats, ys, length):
    taus = range(0, 5000, 2)
    points = []
    for tau in taus:
        num_correct = 0
        num_alarm, cur_y_hats = len(y_hats), y_hats
        num_truth, cur_ys = len(ys), ys
        for y_hat in cur_y_hats:
            y_closest = min((abs(y - y_hat), y) for y in cur_ys)[1]
            y_back = min((abs(y - y_closest), y) for y in cur_y_hats)[1]
            if y_back == y_hat and abs(y_hat - y_closest) <= tau:
                num_correct += 1
        TPR = num_correct * 1. / num_truth
        FPR = 1 - num_correct * 1. / num_alarm
        print("tau={}, num_correct={}, len_y={}, len_y_hat={}, TPR={}, FPR={}".format(
            tau, num_correct, len(ys), len(y_hats), TPR, FPR))
        points.append((FPR, TPR))
    points = sorted(points)
    plt.clf()
    plt.plot([p[0] for p in points], [p[1] for p in points], '.-')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig("./img/roc.png")


# draw ROC curve by varying distance threshold tau
def draw_roc_threshold(y_hats, ys, dists):
    taus = sorted([dists[i] for i in y_hats]) + [1.0]
    tolerance = 200
    points = []
    for tau in taus:
        num_correct = 0
        cur_y_hats = [y for y in y_hats if dists[y] >= tau]
        num_truth = len(ys)
        num_alarm = len(cur_y_hats)
        if num_alarm == 0:
            continue
        for y_hat in cur_y_hats:
            y_closest = min((abs(y - y_hat), y) for y in ys)[1]
            y_back = min((abs(y - y_closest), y) for y in cur_y_hats)[1]
            if y_back == y_hat and abs(y_hat - y_closest) <= tolerance:
                num_correct += 1
        TPR = num_correct * 1. / num_truth
        FPR = 1 - num_correct * 1. / num_alarm
        print("tau={}, num_correct={}, len_y={}, len_y_hat={}, cur_len_y_hat={}, TPR={}, FPR={}".format(
            tau, num_correct, len(ys), len(y_hats), num_alarm, TPR, FPR))
        points.append((FPR, TPR))
    points = sorted(points)
    plt.clf()
    plt.plot([p[0] for p in points], [p[1] for p in points], '.-')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig("./img/roc.png")
