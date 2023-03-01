def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0

    for x, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(x).argmax(axis=1) == y).sum().asscalar()
        n += y.size

        return acc_sum / n