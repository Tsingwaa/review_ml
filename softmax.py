def softmax(x):
    import numpy as np
    x = np.array(x)
    # x = x - np.max(x, axis=1)

    return [np.exp(i) / np.exp(i).sum() for i in x]


def one_hot(targets, num_classes):
    import numpy as np
    N = targets.shape[0]

    one_hot = np.zeros(N, num_classes)
    for i in range(N):
        one_hot[i][targets[i]] = 1

    return one_hot


def cross_entropy(out, targets):
    out_list = out.numpy().tolist()
    
    pass
