import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(0)

M = 5
H = [64, 64, 32]
LR = 0.12
EPOCHS = 3500
BATCH = 256
TRAIN_SAMPLES = 30000
TEST_SAMPLES = 300000

REST_TIME = 0.01
PLOT_EVERY = 10

def one_hot(k, K):
    v = np.zeros(K, dtype=np.float32)
    v[k] = 1.0
    return v

def build_dataset(n_samples):
    xs = np.random.randint(0, M, size=n_samples)
    ys = np.random.randint(0, M, size=n_samples)
    X = np.zeros((n_samples, 2 * M), dtype=np.float32)
    Y = np.zeros((n_samples, M), dtype=np.float32)
    for i in range(n_samples):
        X[i, :M] = one_hot(xs[i], M)
        X[i, M:] = one_hot(ys[i], M)
        Y[i] = one_hot((xs[i] + ys[i]) % M, M)
    return X, Y

X_train, Y_train = build_dataset(TRAIN_SAMPLES)
X_test, Y_test = build_dataset(TEST_SAMPLES)

def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return (x > 0.0).astype(np.float32)

def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy(probs, Yb):
    eps = 1e-9
    return -np.mean(np.sum(Yb * np.log(probs + eps), axis=1))

def accuracy(probs, Yb):
    return np.mean(np.argmax(probs, axis=1) == np.argmax(Yb, axis=1))

layer_sizes = [2 * M] + H + [M]

Ws = []
bs = []
for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
    scale = np.sqrt(2.0 / in_dim)
    Ws.append((np.random.randn(in_dim, out_dim).astype(np.float32) * scale).astype(np.float32))
    bs.append(np.zeros((1, out_dim), dtype=np.float32))

def forward(Xb):
    zs = []
    hs = [Xb]
    for i in range(len(Ws) - 1):
        z = hs[-1] @ Ws[i] + bs[i]
        h = relu(z)
        zs.append(z)
        hs.append(h)
    logits = hs[-1] @ Ws[-1] + bs[-1]
    probs = softmax(logits)
    return zs, hs, logits, probs

plt.ion()
fig, ax = plt.subplots()
train_loss_history = []
val_acc_history = []
loss_line, = ax.plot([], [])
acc_line, = ax.plot([], [])
ax.set_xlabel("Epoch")
ax.set_title("Training Loss and Validation Accuracy")
ax.legend(["Train Loss", "Validation Accuracy"])

for epoch in range(1, EPOCHS + 1):
    idx = np.random.randint(0, TRAIN_SAMPLES, size=BATCH)
    Xb, Yb = X_train[idx], Y_train[idx]

    zs, hs, logits, probs = forward(Xb)
    train_loss = cross_entropy(probs, Yb)
    train_loss_history.append(train_loss)

    B = Xb.shape[0]
    dlogits = (probs - Yb) / B

    dWs = [None] * len(Ws)
    dbs = [None] * len(bs)

    dWs[-1] = hs[-1].T @ dlogits
    dbs[-1] = np.sum(dlogits, axis=0, keepdims=True)

    dh = dlogits @ Ws[-1].T
    for i in range(len(Ws) - 2, -1, -1):
        dz = dh * relu_grad(zs[i])
        dWs[i] = hs[i].T @ dz
        dbs[i] = np.sum(dz, axis=0, keepdims=True)
        dh = dz @ Ws[i].T

    for i in range(len(Ws)):
        Ws[i] -= LR * dWs[i]
        bs[i] -= LR * dbs[i]

    if epoch % PLOT_EVERY == 0:
        _, _, _, val_probs = forward(X_test)
        val_acc = accuracy(val_probs, Y_test)
        val_acc_history.append(val_acc)

        loss_line.set_data(range(len(train_loss_history)), train_loss_history)
        acc_line.set_data(
            np.linspace(0, len(train_loss_history), len(val_acc_history)),
            val_acc_history
        )

        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

    if epoch % 500 == 0 or epoch == 1:
        _, _, _, p_test = forward(X_test)
        acc = accuracy(p_test, Y_test)
        print(f"epoch {epoch:4d} | loss {train_loss:.4f} | val-acc {acc*100:.2f}%")

    time.sleep(REST_TIME)

plt.ioff()
plt.show()

def predict(x, y):
    x = int(x) % M
    y = int(y) % M
    inp = np.zeros((1, 2 * M), dtype=np.float32)
    inp[0, :M] = one_hot(x, M)
    inp[0, M:] = one_hot(y, M)
    _, _, _, probs = forward(inp)
    return int(np.argmax(probs, axis=1)[0])

print("\nExample inferences:")
print("2 ⊕ 3 =", predict(2, 3))
print("4 ⊕ 4 =", predict(4, 4))
print("123 ⊕ 987 =", predict(123, 987))
