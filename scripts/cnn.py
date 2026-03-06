import my_engine
import numpy as np
# import numpy as np
import os
import numpy as np
import scipy.signal as signal
from sklearn.base import BaseEstimator, ClassifierMixin
import inspect
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.utils.multiclass import unique_labels
from typing import Annotated, Tuple, List, Union
Tensor = np.ndarray


class BaseNeuralNet(BaseEstimator, ClassifierMixin):
    def __init__(self, eta=0.1, max_iter=2000, batch_size=32, random_state=None, clip_value=5 ,decay=1e-4,
                 early_stopping=False, patience=30, tol=1e-6, validation_fraction=0.1, val_jump =1,
                 verbose=0, activation='relu', task = 'classification', metric ='accuracy'):
        self.eta = eta
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.decay = decay
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        self.activation = activation
        self.tol = tol
        self.validation_fraction = validation_fraction
        self .val_jump = val_jump
        self.weights_ = []
        self.biases_ = []
        self.losses_ = []
        self.val_losses_ = []
        self.val_scores_ =[]
        self.scores_ =[]
        self.best_val_loss_ = float(np.inf)
        self.epochs_no_improve_ = 0
        self.task = task
        self.metric = metric
        self.clip_value = clip_value

        self._init_activation_functions()

    def get_params(self, deep=True):
        """
        Merge child and parent parameters to be fully visible to scikit-learn.
        """
        ch_params = super().get_params(deep)
        base_sig = inspect.signature(BaseNeuralNet.__init__)

        for name, param in base_sig.parameters.items():
            if name == 'self' or param.kind == param.VAR_KEYWORD: continue
            # add to dict if passed via **kwargs:
            if hasattr(self, name):
                ch_params[name] = getattr(self, name)

        return ch_params


    def _init_activation_functions(self):
        """
        Derivatives are calculated w.r.t 'a' (output).
        """
        self.activation_functions_ = {
            'relu': (self._relu, self._relu_deriv),
            'sigmoid': (self._sigmoid, self._sigmoid_deriv),
            'tanh': (self._tanh, self._tanh_deriv),
            'leaky_relu': (self._leaky_relu, self._leaky_relu_deriv),
            'softplus': (self._softplus, self._softplus_deriv),
            'identity': (lambda z: z, lambda a: np.ones_like(a)),
            'softmax': (self._softmax, None)
        }

    def _relu(self, z): return np.maximum(0, z)
    def _relu_deriv(self, a): return np.where(a > 0, 1, 0)
    def _leaky_relu(self, z): return np.where(z > 0, z, 0.01 * z)
    def _leaky_relu_deriv(self, a): return np.where(a > 0, 1, 0.01)
    def _sigmoid(self, z): return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    def _sigmoid_deriv(self, a): return a * (1 - a)
    def _tanh(self, z): return np.tanh(z)
    def _tanh_deriv(self, a): return 1 - a ** 2
    def _softplus(self, z): return np.log(1 + np.exp(np.clip(z, -250, 250)))
    def _softplus_deriv(self, a): return 1 - np.exp(-a)  # 1 - 1/e^a = 1 - 1/(1+e^z) = sigmoid(z)
    def _softmax(self, z):
        ex = np.exp(z - np.max(z, axis=1, keepdims=True))
        return ex / np.sum(ex, axis=1, keepdims=True)

    def _activate(self, z, name):
        if name not in self.activation_functions_: raise ValueError(f"unknown activation: {name}")
        return self.activation_functions_[name][0](z)

    def _deriv(self, a, name):
        if name not in self.activation_functions_: raise ValueError(f"unknown activation: {name}")
        if self.activation_functions_[name][1] is None:
            # Softmax uses "log‑sum‑exp" trick; then propagate back and calculates (yhat - y)
            # for each component without computing the Jacobian matrix (off-diag contributions vanish on
            # one-hot encoded y); the same update trick also applies to 0.5 * MSE/identity for regression
            # tasks <polymorphic consistent> because of the canonical link between logit and identity
            raise ValueError(f"derivative for {name} is handled in Loss/Jacobian.")
        return self.activation_functions_[name][1](a)

    def _initialize_architecture(self, n_features, n_classes):
        raise NotImplementedError("subclasses must implement _initialize_architecture()")

    def _forward(self, X):
        raise NotImplementedError("subclasses must implement _forward()")

    def _get_gradients(self, X, y, zs, acts):
        raise NotImplementedError("subclasses must implement _get_gradients()")

    def _one_hot(self, y, n_classes):
        return np.eye(n_classes)[y]

    def _score_metric(self, y_true, y_pred):
        if y_true.ndim > 1:
            y_t = np.argmax(y_true, axis=1)
        else:
            y_t = y_true
        y_p = np.argmax(y_pred, axis=1)

        if self.metric == 'accuracy':
            return np.mean(y_t == y_p)

        n_classes = self.n_classes_
        true = (y_t[:, None] == np.arange(n_classes))
        pred = (y_p[:, None] == np.arange(n_classes))

        tp = np.sum(true & pred, axis=0)
        fp = np.sum((~true) & pred, axis=0)
        fn = np.sum(true & (~pred), axis=0)
        epsilon = 1e-7

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)

        w = np.sum(true, axis=0)
        tot_w = np.sum(w)

        if self.metric == 'precision':
            return np.sum(precision * w) / tot_w
        elif self.metric == 'recall':
            return np.sum(recall * w) / tot_w
        elif self.metric == 'f1':
            return np.sum(f1 * w) / tot_w
        return 0.0


    def _setup_task(self, y):
        if self.task == 'classification':
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            y_mapped = np.searchsorted(self.classes_, y)
            y_proc = self._one_hot(y_mapped, self.n_classes_)

            final_act = 'softmax'
            self._loss_func = self.multinomial_cross_entropy

        elif self.task == 'regression':
            # make y shape [N, Outputs]
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            y_proc = np.array(y)
            self.n_classes_ = y.shape[1]  # n_outputs for regression

            final_act = 'identity'
            self._loss_func = self.mean_squared_error

        else:
            raise ValueError(f"Unknown task: {self.task}")

        return y_proc, final_act
    def multinomial_cross_entropy(self, y_true, y_pred):
        # y_true is one_hot_encoded
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def mean_squared_error(self, y_true, y_pred):
        # 0.5 * MSE makes the gradient exactly (pred - y)
        return 0.5 * np.mean((y_true - y_pred) ** 2).get()

    def _validate_epoch(self, X, y, epoch):
        proba = self._forward(X)[1][-1]
        loss = self._loss_func(y, proba)
        self.val_losses_.append(loss)
        score = self._score_metric(y, proba)

        if not hasattr(self, 'val_scores_'): self.val_scores_ = []
        self.val_scores_.append(score)

        if loss < self.best_val_loss_:
            self.best_val_loss_ = loss
            self.best_epoch_ = epoch + 1
            if self.early_stopping:
                self.epochs_no_improve_ = 0
                self.best_weights_ = [w.copy() for w in self.weights_]
                self.best_biases_ = [b.copy() for b in self.biases_]
        else:
            if self.early_stopping: self.epochs_no_improve_ += 1

        if self.early_stopping and self.epochs_no_improve_ >= self.patience:
            self.weights_ = self.best_weights_
            self.biases_ = self.best_biases_
            print(f"Early stopping at epoch {epoch + 1}")
            return True
        return False

    def _process_batch(self, X, y, epoch):
        #Forward
        zs, acts = self._forward(X)
        # Backward
        Gw, Gb = self._get_gradients(X, y, zs, acts)
        # Update
        lr = self.eta / (1 + self.decay * epoch)
        for i in range(len(self.weights_)):
            gw = np.clip(Gw[i], -self.clip_value, self.clip_value)
            gb = np.clip(Gb[i], -self.clip_value, self.clip_value)
            self.weights_[i] -= lr * gw
            self.biases_[i] -= lr * gb
        return self._loss_func(y, acts[-1]), acts[-1]

    def fit(self, XX, yy):
        if self.task == 'classification':
            XX, yy = check_X_y(XX, yy, allow_nd=True)
        else:
            XX = check_array(XX, allow_nd=True)
            yy = np.array(yy)
        XX = np.array(XX)
        y_proc, self.final_activation_ = self._setup_task(yy)

        # self.n_samples_, self.n_features_ = XX.shape
        if XX.ndim == 2:
            self.n_samples_, self.n_features_ = XX.shape
        else:
            self.n_samples_ = XX.shape[0]
            self.n_features_ = int(np.prod(XX.shape[1:]))
        self._initialize_architecture(self.n_features_, y_proc.shape[1])

        rgen = np.random.RandomState(self.random_state)
        if self.validation_fraction and self.validation_fraction > 0:
            val_n = int(self.validation_fraction * self.n_samples_)
            perm = rgen.permutation(self.n_samples_)
            X_val, XX = XX[perm][:val_n], XX[perm][val_n:]
            y_val_enc, y_proc = y_proc[perm][:val_n], y_proc[perm][val_n:]
            self.n_samples_ = XX.shape[0]
            self.val_losses_ = []
        else:
            X_val = y_val_enc = None

        if self.batch_size is None: self.batch_size = 32
        self.activation = str.lower(self.activation)

        for epoch in range(self.max_iter):
            idx = rgen.permutation(self.n_samples_).astype(int)
            X_sh, y_sh = XX[idx], y_proc[idx]
            for l in range(0, self.n_samples_, self.batch_size):
                r = l + self.batch_size
                X, y = X_sh[l:r], y_sh[l:r]

                loss, pred = self._process_batch(X, y, epoch)
                score = self._score_metric(y, pred)
            self.losses_.append(loss)
            self.scores_.append(score)
            if X_val is not None and (epoch % self.val_jump == 0 or epoch == self.max_iter - 1):
                if self._validate_epoch(X_val, y_val_enc, epoch): break

            if self.verbose > 0 and (epoch % self.verbose == 0 or epoch == self.max_iter - 1):
                print(f'epoch {epoch:4d} loss {loss:.5f}', end='')
                if self.val_losses_:
                    print(f' - val_loss {self.val_losses_[-1]:.5f}')
                else:
                    print('')
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X, allow_nd=True)

        n_samples = X.shape[0]
        results = []
        for l in range(0, n_samples, self.batch_size):
            r = l + self.batch_size
            X_batch = np.array(X[l:r])
            acts = self._forward(X_batch)[1]
            batch_probs = acts[-1]
            results.append(batch_probs)
            del X_batch
            del acts
        return np.concatenate(results, axis=0)

    def predict(self, X):
        scores = self.predict_proba(X)
        if self.task == 'regression':
            return scores
        return np.argmax(scores, axis=1)


class NeuralNetwork(BaseNeuralNet):
    def __init__(self, layers=[100], **kwargs):
        super().__init__(**kwargs)
        self.layers = layers

    def _initialize_architecture(self, n_features, n_outputs):
        """
        initializes MLP specific weights (He Init)
        """
        rgen = np.random.RandomState(self.random_state)
        self.layer_sizes_ = [n_features] + self.layers + [n_outputs]
        self.n_layers_ = len(self.layer_sizes_)
        self.weights_ = []
        self.biases_ = []

        for i in range(self.n_layers_ - 1):
            if(self.activation == 'relu' or self.activation == 'leaky_relu'):
                # He Initialization
                std = np.sqrt(2.0 / self.layer_sizes_[i])
            else:
                # Xavier Initialization
                std = np.sqrt(2.0 / (self.layer_sizes_[i] + self.layer_sizes_[i + 1]))
            w = rgen.normal(0.0, scale=std, size=(self.layer_sizes_[i], self.layer_sizes_[i + 1]))
            b = np.zeros((1, self.layer_sizes_[i + 1]))
            self.weights_.append(w)
            self.biases_.append(b)

        self.best_weights_ = [w.copy() for w in self.weights_]
        self.best_biases_ = [b.copy() for b in self.biases_]


    def _forward(self, X):
        zs = []
        acts = [X]
        cur = X

        for i in range(len(self.weights_)):
            z = cur @ self.weights_[i] + self.biases_[i]
            zs.append(z)
            if i == len(self.weights_) - 1:
                a = self._activate(z, self.final_activation_)
            else:
                a = self._activate(z, self.activation)
            acts.append(a)
            cur = a

        return zs, acts

    def _get_gradients(self, X, y, zs, acts):
        """
        calculates gradients for MLP (Backprop)
        """
        Gw = [None] * len(self.weights_)
        Gb = [None] * len(self.biases_)

        d= acts[-1] - y
        # classification(ce + softmax)/regression(mse + I) implements the Jacobian trick,
        # mirroring canonical link property [logit - identity] yielding polymorphic consistent gradient

        Gw[-1] = acts[-2].T @ d / X.shape[0]
        Gb[-1] = np.mean(d, axis=0, keepdims=True)

        for i in range(len(self.weights_) - 2, -1, -1):
            d = d @ self.weights_[i + 1].T * self._deriv(acts[i + 1], self.activation)
            Gw[i] = acts[i].T @ d / X.shape[0]
            Gb[i] = np.mean(d, axis=0, keepdims=True)

        return Gw, Gb
class FastConvNet(BaseNeuralNet):
    def __init__(self, n_filters=8, kernel_size=3, pool_size=2, dense_layers=[100], **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dense_layers = dense_layers

    def _initialize_architecture(self, input_shape, n_outputs):
        # input_shape is (C, H, W)
        if isinstance(input_shape, (int, np.integer)):
            if hasattr(self, 'input_shape_'):
                input_shape = self.input_shape_
            else:
                raise ValueError("FastConvNet requires a 4D input (N, C, H, W), but got scalar input.")
        C, H, W = input_shape

        # 1. Weights (He Init) - Stored on CPU for the Engine, copied to GPU for updates
        std_conv = np.sqrt(2.0 / (C * self.kernel_size * self.kernel_size))

        # Shape: (Filters, Channels, K, K)
        self.conv_W_ = np.random.normal(0.0, std_conv,
                                        (self.n_filters, C, self.kernel_size, self.kernel_size)).astype(np.float32)
        self.conv_b_ = np.zeros((self.n_filters, 1), dtype=np.float32)

        # 2. Dimensions
        self.conv_H = H
        self.conv_W = W

        self.pool_H = self.conv_H // self.pool_size
        self.pool_W = self.conv_W // self.pool_size

        # 3. Dense Layers Setup
        flat_input_size = self.n_filters * self.pool_H * self.pool_W
        self.layer_sizes_ = [flat_input_size] + self.dense_layers + [n_outputs]

        # Initialize Dense Weights (Standard MLP logic)
        self.weights_ = []
        self.biases_ = []
        rgen = np.random.RandomState(self.random_state)

        for i in range(len(self.layer_sizes_) - 1):
            std = np.sqrt(2.0 / self.layer_sizes_[i])
            w = rgen.normal(0.0, scale=std, size=(self.layer_sizes_[i], self.layer_sizes_[i + 1]))
            b = np.zeros((1, self.layer_sizes_[i + 1]))
            self.weights_.append(w)
            self.biases_.append(b)

    def _forward(self, X):
        # X is (N, C, H, W) on GPU (CuPy)
        # We need to move to CPU for your Engine
        X_cpu = X.astype(np.float32)
        N, C, H, W = X_cpu.shape

        # --- 1. C++ FAST CONVOLUTION ---
        # Output: (N, Filters, H_out, W_out)
        conv_out = np.zeros((N, self.n_filters, self.conv_H, self.conv_W), dtype=np.float32)

        # Your engine processes 1 image at a time (currently). 
        # We loop over the batch. This is still faster than pure Python loops.
        for i in range(N):
            conv_out[i] = my_engine.Hazem_Convolution(X_cpu[i], self.conv_W_, 1)

            # FIX: Reshape bias to (Filters, 1, 1) so it broadcasts over (Filters, H, W)
            conv_out[i] += self.conv_b_.reshape(-1, 1, 1)

        # Move back to GPU for Activation & Pooling
        conv_out_gpu = np.array(conv_out)
        self.conv_z_ = conv_out_gpu
        self.conv_a_ = self._activate(conv_out_gpu, self.activation)

        # --- 2. MAX POOLING (CuPy is fast at this) ---
        N, F, H_c, W_c = self.conv_a_.shape
        reshaped = self.conv_a_.reshape(N, F, H_c//self.pool_size, self.pool_size, W_c//self.pool_size, self.pool_size)
        pool_out = reshaped.max(axis=(3, 5))

        # Save mask for backprop
        self.pool_mask_ = (reshaped == pool_out[:, :, :, None, :, None])
        self.pool_a_ = pool_out

        # --- 3. FLATTEN & DENSE ---
        flat = pool_out.reshape(N, -1)

        # Standard MLP Forward
        zs = []
        acts = [flat]
        cur = flat
        for i in range(len(self.weights_)):
            z = cur @ self.weights_[i] + self.biases_[i]
            zs.append(z)
            act_func = self.final_activation_ if i == len(self.weights_) - 1 else self.activation
            a = self._activate(z, act_func)
            acts.append(a)
            cur = a

        return zs, acts

    def _get_gradients(self, X, y, zs, acts):
        # --- 1. DENSE BACKPROP ---
        Gw = [None] * len(self.weights_)
        Gb = [None] * len(self.biases_)
        d = acts[-1] - y
        Gw[-1] = acts[-2].T @ d / X.shape[0]
        Gb[-1] = np.mean(d, axis=0, keepdims=True)
        for i in range(len(self.weights_) - 2, -1, -1):
            d = d @ self.weights_[i + 1].T * self._deriv(acts[i + 1], self.activation)
            Gw[i] = acts[i].T @ d / X.shape[0]
            Gb[i] = np.mean(d, axis=0, keepdims=True)
        d_flat = d @ self.weights_[0].T * self._deriv(acts[0], self.activation)

        # --- 2. UNPOOLING (FIXED) ---
        d_pool = d_flat.reshape(self.pool_a_.shape)
        d_conv_act = np.zeros_like(self.conv_a_)
        N, F, H_p, W_p = d_pool.shape

        # 1. Create 6D container
        d_reshaped = d_conv_act.reshape(N, F, H_p, self.pool_size, W_p, self.pool_size)

        # 2. Expand 4D gradient to full size
        d_upsampled_4d = d_pool.repeat(self.pool_size, axis=2).repeat(self.pool_size, axis=3)

        # 3. FIX: Reshape expanded gradient to 6D so it matches the mask
        d_upsampled_6d = d_upsampled_4d.reshape(d_reshaped.shape)

        # 4. Apply 6D Mask to 6D Gradient
        d_reshaped[self.pool_mask_] = d_upsampled_6d[self.pool_mask_]

        d_conv_act = d_reshaped.reshape(self.conv_a_.shape)
        d_conv_z = d_conv_act * self._deriv(self.conv_a_, self.activation)

        # --- 3. CONV BACKPROP ---
        conv_Gw = np.zeros_like(self.conv_W_) # CPU
        conv_Gb = np.sum(d_conv_z, axis=(0, 2, 3)).reshape(self.n_filters, 1) / N

        X_cpu = X
        d_z_cpu = d_conv_z

        pad = (self.kernel_size - 1) // 2
        X_padded = np.pad(X_cpu, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')

        for n in range(N):
            for f in range(self.n_filters):
                for c in range(X.shape[1]):
                    # Valid Correlation between Padded Input and Error
                    conv_Gw[f, c] += signal.correlate2d(X_padded[n, c], d_z_cpu[n, f], mode='valid')

        conv_Gw /= N

        self.conv_Gw_ = np.array(conv_Gw)
        self.conv_Gb_ = conv_Gb

        return Gw, Gb

    def _process_batch(self, X, y, epoch):
        loss, pred = super()._process_batch(X, y, epoch)

        # Update Convolutional Weights
        lr = self.eta / (1 + self.decay * epoch)

        # Move CPU weights to GPU for update math, then back? 
        # Actually easier to just update the CPU master copy using GPU gradients
        gw = np.clip(self.conv_Gw_, -self.clip_value, self.clip_value)
        gb = np.clip(self.conv_Gb_, -self.clip_value, self.clip_value)

        self.conv_W_ -= lr * gw
        self.conv_b_ -= lr * gb.reshape(self.n_filters, 1)

        return loss, pred

    def fit(self, XX, yy):
        # Image check
        if hasattr(XX, 'shape') and len(XX.shape) == 4:
            self.input_shape_ = XX.shape[1:]

            # Pass control to BaseNeuralNet.fit, which handles the loop and validation
        return super().fit(XX, yy)
if __name__ == "__main__":
    # Fake Data: 10 images, 1 channel (Grayscale), 28x28
    X_train = np.random.rand(10, 1, 28, 28).astype(np.float32)
    y_train = np.random.randint(0, 2, size=10) # Binary classification

    print("Initializing C++ Hybrid CNN...")
    # Use your custom engine class
    cnn = FastConvNet(max_iter=5, verbose=1, n_filters=4)

    print("Training...")
    cnn.fit(X_train, y_train)

    print("Inference Check (Using C++ Engine):")
    preds = cnn.predict(X_train[:2])
    print(f"Predictions: {preds}")
    print("✅🐢 Integration Successful.")